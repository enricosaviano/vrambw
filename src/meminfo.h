#pragma once
#include <string>
#include <cstdint>

// Heuristic description of the installed system RAM, read from SMBIOS
// (raw table if readable, i.e. root) or from the udev database (no root needed).
struct MemInfo {
    bool ok = false;           // something useful was found
    std::string source;        // "SMBIOS" or "udev"
    std::string type;          // e.g. "DDR4", "LPDDR5/5X"; empty if unknown
    bool type_guessed = false; // type inferred from voltage/speed
    int mts = 0;               // configured speed, MT/s
    int bus_bits = 0;          // total data bus width
    int channels = 0;          // physical channels found
    int chan_bits = 0;         // width of each channel (0 if not uniform)
    bool chan_guessed = false; // channels not named in slot labels: 1 module = 1 channel
    bool ecc = false;          // modules have ECC bits (not counted in bus_bits)
    bool chan_desktop = false; // no channel labels, <= 4 slots: assumed dual-channel
    int slots = 0;             // memory slots on the board (populated + empty)
    int sockets = 0;
    int modules = 0;
    uint64_t size = 0;         // bytes
    double peak_gbs = 0.0;     // mts * bus_bits / 8, GB/s
    std::string reason;        // why nothing was found
};

MemInfo detect_memory();

// One-line description, e.g. "LPDDR5/5X-6400, 4 x 32-bit = 128-bit (dual-channel equivalent), 16 GiB"
std::string describe_memory(const MemInfo &m);
