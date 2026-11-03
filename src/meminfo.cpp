// vrambw - heuristic detection of system RAM type and channel layout.
//
// Sources, in order:
//   1. raw SMBIOS table /sys/firmware/dmi/tables/DMI (readable by root only)
//   2. udev database /run/udev/data/+dmi:id (world readable, systemd >= 248)
// SMBIOS "Memory Device" (type 17) records give type, speed, width, voltage
// and slot labels. Type is guessed from voltage/speed when missing or unknown
// to the firmware/udev tables. Channels are counted by grouping modules on
// socket + controller + channel parsed from the slot labels.
#include "meminfo.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

namespace {

struct Dev {
    int code = 0;       // SMBIOS memory type code (raw table only)
    std::string tname;  // type name (udev only)
    int speed = 0;      // rated MT/s
    int cspeed = 0;     // configured MT/s
    int dbits = 0;      // data width (some BIOSes wrongly include ECC bits)
    int tbits = 0;      // total width, including ECC
    uint64_t size = 0;  // bytes
    int form = 0;       // SMBIOS form factor code
    std::string loc, bank;
    double volt = 0.0;  // configured (or minimum) voltage
};

const int FORM_DIMM = 0x09, FORM_CHIPS = 0x0B, FORM_SODIMM = 0x0D;

const char *code_name(int c)
{
    switch (c) {
    case 0x0F: return "SDRAM";
    case 0x12: return "DDR";
    case 0x13: return "DDR2";
    case 0x18: return "DDR3";
    case 0x1A: return "DDR4";
    case 0x1B: return "LPDDR";
    case 0x1C: return "LPDDR2";
    case 0x1D: return "LPDDR3";
    case 0x1E: return "LPDDR4";
    case 0x20: return "HBM";
    case 0x21: return "HBM2";
    case 0x22: return "DDR5";
    case 0x23: return "LPDDR5";
    case 0x24: return "HBM3";
    }
    return "";
}

// ---------------------------------------------------------------- raw SMBIOS
bool read_smbios(std::vector<Dev> &out, int &slots)
{
    const char *path = getenv("VRAMBW_DMI"); // override for testing
    FILE *f = fopen(path ? path : "/sys/firmware/dmi/tables/DMI", "rb");
    if (!f) return false;
    std::vector<unsigned char> b;
    unsigned char tmp[4096];
    size_t n;
    while ((n = fread(tmp, 1, sizeof(tmp), f)) > 0) b.insert(b.end(), tmp, tmp + n);
    fclose(f);

    size_t p = 0, sz = b.size();
    while (p + 4 <= sz) {
        int type = b[p], len = b[p + 1];
        if (len < 4 || p + len > sz) break;

        // string set: NUL-terminated strings, ended by an extra NUL
        std::vector<std::string> strs;
        size_t q = p + len;
        if (q + 1 < sz && b[q] == 0 && b[q + 1] == 0) {
            q += 2;
        } else {
            while (q < sz) {
                size_t s = q;
                while (q < sz && b[q]) q++;
                strs.emplace_back((const char *)&b[s], q - s);
                q++;
                if (q < sz && b[q] == 0) { q++; break; }
            }
        }

        if (type == 17 && len >= 0x15) {
            const unsigned char *r = &b[p];
            auto w16 = [&](int o) { return o + 2 <= len ? (int)(r[o] | r[o + 1] << 8) : 0; };
            auto w32 = [&](int o) {
                return o + 4 <= len ? (uint32_t)r[o] | (uint32_t)r[o + 1] << 8 |
                                          (uint32_t)r[o + 2] << 16 | (uint32_t)r[o + 3] << 24
                                    : 0u;
            };
            auto str = [&](int o) {
                int i = o < len ? r[o] : 0;
                return (i > 0 && i <= (int)strs.size()) ? strs[i - 1] : std::string();
            };

            int s = w16(0x0C);
            slots++;
            if (s != 0) { // 0 = empty slot
                Dev d;
                if (s == 0x7FFF) d.size = (uint64_t)w32(0x1C) << 20;
                else if (s == 0xFFFF) d.size = 0;
                else if (s & 0x8000) d.size = (uint64_t)(s & 0x7FFF) << 10;
                else d.size = (uint64_t)s << 20;
                d.tbits = w16(0x08);
                if (d.tbits == 0xFFFF) d.tbits = 0;
                d.dbits = w16(0x0A);
                if (d.dbits == 0xFFFF) d.dbits = 0;
                d.form = r[0x0E];
                d.loc = str(0x10);
                d.bank = str(0x11);
                d.code = r[0x12];
                d.speed = w16(0x15);
                if (d.speed == 0xFFFF) d.speed = (int)w32(0x54);
                d.cspeed = w16(0x20);
                if (d.cspeed == 0xFFFF) d.cspeed = (int)w32(0x58);
                int mv = w16(0x26);
                if (!mv) mv = w16(0x22);
                d.volt = mv / 1000.0;
                out.push_back(d);
            }
        }
        if (type == 127) break;
        p = q;
    }
    return !out.empty();
}

// ---------------------------------------------------------------- udev db
bool read_udev(std::vector<Dev> &out, int &slots)
{
    const char *path = getenv("VRAMBW_UDEV_DB"); // override for testing
    FILE *f = fopen(path ? path : "/run/udev/data/+dmi:id", "r");
    if (!f) return false;
    std::map<long, std::map<std::string, std::string>> m;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (!strncmp(line, "E:MEMORY_ARRAY_NUM_DEVICES=", 27)) slots = std::max(slots, atoi(line + 27));
        if (strncmp(line, "E:MEMORY_DEVICE_", 16)) continue;
        char *p = line + 16, *e;
        long idx = strtol(p, &e, 10);
        if (e == p || *e != '_') continue;
        char *key = e + 1, *eq = strchr(key, '=');
        if (!eq) continue;
        *eq = 0;
        char *val = eq + 1;
        val[strcspn(val, "\r\n")] = 0;
        m[idx][key] = val;
    }
    fclose(f);

    slots = std::max(slots, (int)m.size()); // includes empty slots (PRESENT=0)
    for (auto &kv : m) {
        auto &v = kv.second;
        auto get = [&](const char *k) { auto it = v.find(k); return it == v.end() ? std::string() : it->second; };
        if (get("PRESENT") == "0") continue;
        Dev d;
        d.size = strtoull(get("SIZE").c_str(), nullptr, 10);
        if (!d.size) continue;
        d.tname = get("TYPE");
        d.speed = atoi(get("SPEED_MTS").c_str());
        d.cspeed = atoi(get("CONFIGURED_SPEED_MTS").c_str());
        d.dbits = atoi(get("DATA_WIDTH").c_str());
        d.tbits = atoi(get("TOTAL_WIDTH").c_str());
        std::string ff = get("FORM_FACTOR");
        d.form = ff == "Row Of Chips" ? FORM_CHIPS : ff == "SODIMM" ? FORM_SODIMM : ff == "DIMM" ? FORM_DIMM : 0;
        d.loc = get("LOCATOR");
        d.bank = get("BANK_LOCATOR");
        std::string vs = get("CONFIGURED_VOLTAGE");
        if (vs.empty()) vs = get("MINIMUM_VOLTAGE");
        d.volt = atof(vs.c_str());
        out.push_back(d);
    }
    return !out.empty();
}

// ---------------------------------------------------------------- type
std::string reported_type(const Dev &d)
{
    if (d.code) return code_name(d.code);
    const std::string &t = d.tname;
    if (!t.compare(0, 3, "DDR") || !t.compare(0, 5, "LPDDR") || !t.compare(0, 3, "HBM")) return t;
    return ""; // "<OUT OF SPEC>", "Unknown", "Other"...
}

std::string guess_type(const Dev &d, int mts)
{
    double v = d.volt;
    bool chips = d.form == FORM_CHIPS;
    if (v > 0.40 && v < 0.55) return "LPDDR5";
    if (v >= 0.55 && v < 0.70) return "LPDDR4X";
    if (v >= 1.05 && v < 1.15) return (chips && mts && mts <= 4266) ? "LPDDR4" : "DDR5";
    if (v >= 1.15 && v < 1.30) return "DDR4";
    if (v >= 1.30 && v < 1.45) return "DDR3L";
    if (v >= 1.45 && v < 1.60) return "DDR3";
    if (mts > 0) {
        if (chips && mts >= 5500) return "LPDDR5";
        if (chips && mts >= 3733 && mts <= 4266) return "LPDDR4X";
        if (mts >= 4400) return "DDR5";
        if (mts >= 2133) return "DDR4";
        if (mts >= 800) return "DDR3";
    }
    return "";
}

// SMBIOS has no codes for LPDDR4X / LPDDR5X / DDR3L: refine with voltage and speed
std::string refine_type(const std::string &t, const Dev &d, int mts)
{
    if (t == "LPDDR4" && ((d.volt >= 0.55 && d.volt < 0.70) || mts > 4266)) return "LPDDR4X";
    if (t == "LPDDR5") {
        if (mts > 6400) return "LPDDR5X";
        if (mts == 6400) return "LPDDR5/5X"; // top LPDDR5 speed, also a common LPDDR5X speed
    }
    if (t == "DDR3" && d.volt >= 1.30 && d.volt < 1.45) return "DDR3L";
    return t;
}

// ---------------------------------------------------------------- channels
std::string upper(std::string s)
{
    for (auto &c : s) c = (char)toupper((unsigned char)c);
    return s;
}

bool word_start(const std::string &s, size_t i) { return i == 0 || !isalnum((unsigned char)s[i - 1]); }

// "<TAG>[ _-]<alnum run>" anywhere in s; tag must start a word
std::string after_tag(const std::string &s, const char *tag, bool digits_only)
{
    size_t tl = strlen(tag);
    for (size_t i = s.find(tag); i != std::string::npos; i = s.find(tag, i + 1)) {
        if (!word_start(s, i)) continue;
        size_t j = i + tl;
        while (j < s.size() && (s[j] == ' ' || s[j] == '_' || s[j] == '-')) j++;
        size_t k = j;
        while (k < s.size() && (digits_only ? isdigit((unsigned char)s[k]) : isalnum((unsigned char)s[k]))) k++;
        if (k > j) {
            std::string r = s.substr(j, k - j);
            // "CHANNELA1": keep the channel letter only
            if (!digits_only && isalpha((unsigned char)r[0]) && r.size() > 1) r = r.substr(0, 1);
            return r;
        }
    }
    return "";
}

std::string socket_of(const std::string &s)
{
    std::string r = after_tag(s, "CPU", true);
    if (r.empty()) r = after_tag(s, "SOCKET", true);
    if (r.empty()) {
        // "P0 CHANNEL A", "P1-DIMMA1", "P0_NODE0_..."
        for (size_t i = 0; i + 1 < s.size(); i++)
            if (s[i] == 'P' && word_start(s, i) && isdigit((unsigned char)s[i + 1])) {
                size_t k = i + 1;
                while (k < s.size() && isdigit((unsigned char)s[k])) k++;
                if (k == s.size() || !isalnum((unsigned char)s[k])) return s.substr(i + 1, k - i - 1);
            }
    }
    return r.empty() ? "0" : r;
}

std::string channel_of(const std::string &s)
{
    std::string r = after_tag(s, "CHANNEL", false);
    if (!r.empty()) return r;
    // "DIMM_A1", "DIMMA1", "DIMM A2": letter followed by slot digit
    for (size_t i = s.find("DIMM"); i != std::string::npos; i = s.find("DIMM", i + 1)) {
        size_t j = i + 4;
        while (j < s.size() && (s[j] == ' ' || s[j] == '_' || s[j] == '-')) j++;
        if (j + 1 < s.size() && isalpha((unsigned char)s[j]) && isdigit((unsigned char)s[j + 1]))
            return std::string(1, s[j]);
    }
    return "";
}

const char *channel_word(int n)
{
    switch (n) {
    case 1: return "single";
    case 2: return "dual";
    case 3: return "triple";
    case 4: return "quad";
    case 6: return "hexa";
    case 8: return "octa";
    }
    return nullptr;
}

} // namespace

MemInfo detect_memory()
{
    MemInfo m;
    std::vector<Dev> devs;
    int slots = 0;
    if (!getenv("VRAMBW_UDEV_DB") && read_smbios(devs, slots)) m.source = "SMBIOS";
    else if ((slots = 0, read_udev(devs, slots))) m.source = "udev";
    else {
        m.reason = "SMBIOS not readable (run as root for details)";
        return m;
    }

    // speed: lowest configured speed (the whole bus runs at it).
    // SMBIOS <= 2.7 BIOSes report the configured speed as clock in MHz (half the
    // MT/s): DDR3-1600 shows 800. Detect it against the rated speed.
    for (auto &d : devs) {
        if (d.cspeed > 0 && d.speed > 0 && d.cspeed < d.speed * 0.6 && d.cspeed * 2 <= d.speed * 1.05)
            d.cspeed *= 2;
        int s = d.cspeed > 0 ? d.cspeed : d.speed;
        if (s > 0 && (m.mts == 0 || s < m.mts)) m.mts = s;
    }

    // type: from the first module that reports one, else guessed
    for (auto &d : devs) {
        std::string t = reported_type(d);
        if (!t.empty()) { m.type = refine_type(t, d, m.mts); break; }
    }
    if (m.type.empty()) {
        m.type = refine_type(guess_type(devs[0], m.mts), devs[0], m.mts);
        m.type_guessed = !m.type.empty();
    }
    bool lp = !m.type.compare(0, 2, "LP");

    // both speeds in MHz on very old BIOSes: below the JEDEC minimum MT/s of the type
    {
        int jedec_min = m.type == "DDR2" ? 400 : !m.type.compare(0, 4, "DDR3") ? 800
                      : m.type == "DDR4" ? 1600 : m.type == "DDR5" ? 3200 : 0;
        if (m.mts > 0 && m.mts < jedec_min) m.mts *= 2;
    }

    // channels: group by socket|controller|channel
    std::map<std::string, int> chan_width;
    std::set<std::string> sockets;
    for (size_t i = 0; i < devs.size(); i++) {
        const Dev &d = devs[i];
        std::string lab = upper(d.bank + " " + d.loc);
        std::string sock = socket_of(lab), ch = channel_of(lab);
        std::string ctl = after_tag(lab, "CONTROLLER", true);
        std::string key;
        if (ch.empty()) {
            m.chan_guessed = true;
            key = "#" + std::to_string(i);
        } else {
            key = sock + "|" + ctl + "|" + ch;
        }
        sockets.insert(sock);
        int w = d.dbits > 0 ? d.dbits : (lp ? 0 : 64);
        // strip ECC bits: 72 -> 64 (DDR3/4 ECC DIMM), 40 -> 32 (DDR5 ECC subchannel), 80 -> 64
        if (w == 40) w = 32;
        else if (w == 80) w = 64;
        else if (w > 0 && w % 72 == 0) w = w / 72 * 64;
        if (w < d.dbits || (d.tbits > 0 && w > 0 && d.tbits > w)) m.ecc = true;
        int &cw = chan_width[key];
        if (w == 0) cw = -1; // unknown width
        else if (cw >= 0) cw = std::max(cw, w);
        m.size += d.size;
    }
    m.modules = (int)devs.size();
    m.sockets = (int)sockets.size();
    m.slots = slots;

    // No slot label names a channel. 1 module per channel is right for servers,
    // but desktop/laptop boards with up to 4 slots are dual-channel (2 slots per channel).
    bool any_labeled = false;
    for (auto &kv : chan_width)
        if (kv.first[0] != '#') any_labeled = true;
    if (!any_labeled && m.sockets == 1 && slots <= 4 && devs.size() > 2) {
        int w = 0;
        for (auto &kv : chan_width) w = (kv.second < 0 || w < 0) ? -1 : std::max(w, kv.second);
        chan_width.clear();
        chan_width["#a"] = w;
        chan_width["#b"] = w;
        m.chan_desktop = true;
    }
    m.channels = (int)chan_width.size();
    bool uniform = true;
    int first = chan_width.begin()->second;
    for (auto &kv : chan_width) {
        if (kv.second < 0) { m.bus_bits = 0; uniform = false; break; }
        m.bus_bits += kv.second;
        if (kv.second != first) uniform = false;
    }
    m.chan_bits = uniform ? first : 0;
    if (m.mts > 0 && m.bus_bits > 0) m.peak_gbs = m.mts * 1e6 * (m.bus_bits / 8.0) / 1e9;
    m.ok = true;
    return m;
}

std::string describe_memory(const MemInfo &m)
{
    if (!m.ok) return "type/channels unknown: " + m.reason;
    char buf[256];
    std::string s;

    if (m.type.empty()) s = "unknown type";
    else s = m.type;
    if (m.mts > 0) { snprintf(buf, sizeof(buf), "-%d", m.mts); s += buf; }
    if (m.type_guessed) s += " (guessed)";

    if (m.bus_bits > 0) {
        if (m.chan_bits > 0) snprintf(buf, sizeof(buf), ", %d x %d-bit = %d-bit", m.channels, m.chan_bits, m.bus_bits);
        else snprintf(buf, sizeof(buf), ", %d-bit bus", m.bus_bits);
        s += buf;
        bool per_socket = m.sockets > 1 && (m.bus_bits / 64) % m.sockets == 0;
        if (m.bus_bits % 64 == 0) {
            int n = m.bus_bits / 64 / (per_socket ? m.sockets : 1);
            const char *w = channel_word(n);
            s += " (";
            if (per_socket) { snprintf(buf, sizeof(buf), "%d sockets x ", m.sockets); s += buf; }
            if (w) snprintf(buf, sizeof(buf), "%s-channel", w);
            else snprintf(buf, sizeof(buf), "%d-channel", n);
            s += buf;
            s += m.chan_bits == 64 ? ")" : " equivalent)";
        }
        if (m.chan_desktop) snprintf(buf, sizeof(buf), " (channels guessed: %d-slot board)", m.slots), s += buf;
        else if (m.chan_guessed) s += " (channels guessed: 1 module per channel)";
        if (m.sockets > 1 && !(per_socket && m.bus_bits % 64 == 0)) {
            snprintf(buf, sizeof(buf), ", %d sockets", m.sockets);
            s += buf;
        }
    } else {
        snprintf(buf, sizeof(buf), ", %d module(s), bus width unknown", m.modules);
        s += buf;
    }
    if (m.ecc) s += ", ECC";
    if (m.size) { snprintf(buf, sizeof(buf), ", %.0f GiB", m.size / 1073741824.0); s += buf; }
    return s;
}
