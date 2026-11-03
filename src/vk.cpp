// vrambw - Vulkan read bandwidth of device-local memory and host RAM.
//
// Same method as the CUDA test: a compute shader streams a buffer much larger
// than the GPU caches, read-only, 16-byte loads; best sample is reported.
// libvulkan.so.1 is opened with dlopen, so the binary runs without Vulkan.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "vk.h"
#include "progress.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

namespace {

const uint32_t SPV[] =
#include "vk_read.spv.inc"
    ;

const uint32_t GROUPS = 8192; // x 256 threads: enough to fill any current GPU

#define FN(name) PFN_##name name = nullptr;
FN(vkGetInstanceProcAddr)
FN(vkCreateInstance)
FN(vkDestroyInstance)
FN(vkEnumeratePhysicalDevices)
FN(vkGetPhysicalDeviceProperties)
FN(vkGetPhysicalDeviceMemoryProperties)
FN(vkGetPhysicalDeviceQueueFamilyProperties)
FN(vkCreateDevice)
FN(vkGetDeviceProcAddr)
FN(vkDestroyDevice)
FN(vkGetDeviceQueue)
FN(vkCreateBuffer)
FN(vkDestroyBuffer)
FN(vkGetBufferMemoryRequirements)
FN(vkAllocateMemory)
FN(vkFreeMemory)
FN(vkBindBufferMemory)
FN(vkCreateShaderModule)
FN(vkDestroyShaderModule)
FN(vkCreateDescriptorSetLayout)
FN(vkDestroyDescriptorSetLayout)
FN(vkCreatePipelineLayout)
FN(vkDestroyPipelineLayout)
FN(vkCreateComputePipelines)
FN(vkDestroyPipeline)
FN(vkCreateDescriptorPool)
FN(vkDestroyDescriptorPool)
FN(vkAllocateDescriptorSets)
FN(vkUpdateDescriptorSets)
FN(vkCreateCommandPool)
FN(vkDestroyCommandPool)
FN(vkAllocateCommandBuffers)
FN(vkResetCommandBuffer)
FN(vkBeginCommandBuffer)
FN(vkEndCommandBuffer)
FN(vkCmdBindPipeline)
FN(vkCmdBindDescriptorSets)
FN(vkCmdPushConstants)
FN(vkCmdDispatch)
FN(vkCmdFillBuffer)
FN(vkQueueSubmit)
FN(vkQueueWaitIdle)
#undef FN

struct Ctx {
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties mem{};
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qf = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkBuffer sink = VK_NULL_HANDLE;
    VkDeviceMemory sink_mem = VK_NULL_HANDLE;
};

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

bool load_device_fns(VkDevice d)
{
#define LD(name) if (!(name = (PFN_##name)vkGetDeviceProcAddr(d, #name))) return false;
    LD(vkDestroyDevice) LD(vkGetDeviceQueue) LD(vkCreateBuffer) LD(vkDestroyBuffer)
    LD(vkGetBufferMemoryRequirements) LD(vkAllocateMemory) LD(vkFreeMemory) LD(vkBindBufferMemory)
    LD(vkCreateShaderModule) LD(vkDestroyShaderModule) LD(vkCreateDescriptorSetLayout)
    LD(vkDestroyDescriptorSetLayout) LD(vkCreatePipelineLayout) LD(vkDestroyPipelineLayout)
    LD(vkCreateComputePipelines) LD(vkDestroyPipeline) LD(vkCreateDescriptorPool)
    LD(vkDestroyDescriptorPool) LD(vkAllocateDescriptorSets) LD(vkUpdateDescriptorSets)
    LD(vkCreateCommandPool) LD(vkDestroyCommandPool) LD(vkAllocateCommandBuffers)
    LD(vkResetCommandBuffer) LD(vkBeginCommandBuffer) LD(vkEndCommandBuffer) LD(vkCmdBindPipeline)
    LD(vkCmdBindDescriptorSets) LD(vkCmdPushConstants) LD(vkCmdDispatch) LD(vkCmdFillBuffer)
    LD(vkQueueSubmit) LD(vkQueueWaitIdle)
#undef LD
    return true;
}

// memory type allowed by `bits` with all `want` flags and none of `avoid`
int find_type(const Ctx &c, uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid)
{
    for (uint32_t i = 0; i < c.mem.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = c.mem.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (f & want) == want && !(f & avoid)) return (int)i;
    }
    return -1;
}

void free_buf(Ctx &c, Buf &b)
{
    if (b.buf) vkDestroyBuffer(c.dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(c.dev, b.mem, nullptr);
    b = Buf();
}

bool alloc_buf(Ctx &c, VkDeviceSize size, int type, Buf &b)
{
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(c.dev, &bi, nullptr, &b.buf) != VK_SUCCESS) { b.buf = VK_NULL_HANDLE; return false; }
    VkMemoryRequirements r;
    vkGetBufferMemoryRequirements(c.dev, b.buf, &r);
    if (!(r.memoryTypeBits & (1u << type))) { free_buf(c, b); return false; }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = r.size;
    ai.memoryTypeIndex = (uint32_t)type;
    if (vkAllocateMemory(c.dev, &ai, nullptr, &b.mem) != VK_SUCCESS) { b.mem = VK_NULL_HANDLE; free_buf(c, b); return false; }
    if (vkBindBufferMemory(c.dev, b.buf, b.mem, 0) != VK_SUCCESS) { free_buf(c, b); return false; }
    b.size = size;
    return true;
}

uint32_t storage_type_bits(Ctx &c)
{
    Buf probe;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = 1 << 20;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(c.dev, &bi, nullptr, &probe.buf) != VK_SUCCESS) return 0;
    VkMemoryRequirements r;
    vkGetBufferMemoryRequirements(c.dev, probe.buf, &r);
    vkDestroyBuffer(c.dev, probe.buf, nullptr);
    return r.memoryTypeBits;
}

double run_cmd(Ctx &c)
{
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c.cmd;
    auto t0 = std::chrono::steady_clock::now();
    if (vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) return -1;
    if (vkQueueWaitIdle(c.queue) != VK_SUCCESS) return -1;
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

bool record(Ctx &c, const Buf &src, int reps, bool fill)
{
    vkResetCommandBuffer(c.cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    if (vkBeginCommandBuffer(c.cmd, &bi) != VK_SUCCESS) return false;
    if (fill) {
        vkCmdFillBuffer(c.cmd, src.buf, 0, VK_WHOLE_SIZE, 0x5a3c96e1u);
    } else {
        uint32_t pc[2] = {(uint32_t)(src.size / 16), 0x7fffffffu};
        vkCmdBindPipeline(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c.pipe);
        vkCmdBindDescriptorSets(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c.pl, 0, 1, &c.set, 0, nullptr);
        vkCmdPushConstants(c.cmd, c.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        uint32_t groups = std::min(GROUPS, c.props.limits.maxComputeWorkGroupCount[0]);
        for (int r = 0; r < reps; r++) vkCmdDispatch(c.cmd, groups, 1, 1);
    }
    return vkEndCommandBuffer(c.cmd) == VK_SUCCESS;
}

// best read bandwidth in GB/s of buffer `src`, <0 on error
double measure(Ctx &c, const Buf &src, int samples, const char *label)
{
    VkDescriptorBufferInfo bi[2] = {{src.buf, 0, VK_WHOLE_SIZE}, {c.sink, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[2];
    for (int i = 0; i < 2; i++) {
        w[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = c.set;
        w[i].dstBinding = (uint32_t)i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(c.dev, 2, w, 0, nullptr);

    progress("  %-13s preparing %llu MiB buffer...", label, (unsigned long long)(src.size >> 20));
    if (!record(c, src, 0, true) || run_cmd(c) < 0) return -1;

    if (!record(c, src, 1, false)) return -1;
    double t1 = run_cmd(c);
    if (t1 < 0) return -1;
    t1 = std::max(t1, 1e-5);
    int reps = std::max(1, std::min(10000, (int)ceil(0.1 / t1))); // ~100 ms per sample
    if (!record(c, src, reps, false)) return -1;

    progress("  %-13s warmup...", label);
    for (double t = 0; t < 0.4;) { // ~0.4 s: lets clocks ramp up
        double d = run_cmd(c);
        if (d < 0) return -1;
        t += d;
    }
    double best = 0;
    for (int s = 0; s < samples; s++) {
        double d = run_cmd(c);
        if (d < 0) return -1;
        double gbs = (double)src.size * reps / d / 1e9;
        best = std::max(best, gbs);
        progress("  %-13s sample %d/%d  %7.1f GB/s   best %7.1f GB/s", label, s + 1, samples, gbs, best);
    }
    progress_clear();
    return best;
}

// allocate the biggest buffer up to 1 GiB (half the heap), shrinking on failure
bool alloc_test_buf(Ctx &c, int type, Buf &b)
{
    VkDeviceSize heap = c.mem.memoryHeaps[c.mem.memoryTypes[type].heapIndex].size;
    VkDeviceSize size = std::min<VkDeviceSize>((VkDeviceSize)1 << 30, heap / 2);
    size = std::min<VkDeviceSize>(size, c.props.limits.maxStorageBufferRange);
    size &= ~(VkDeviceSize)((1 << 20) - 1);
    for (; size >= ((VkDeviceSize)64 << 20); size /= 2)
        if (alloc_buf(c, size, type, b)) return true;
    return false;
}

bool setup(Ctx &c)
{
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qp(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nq, qp.data());
    int qf = -1;
    for (uint32_t i = 0; i < nq; i++) // prefer a compute-only queue, else any with compute
        if ((qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { qf = (int)i; break; }
    if (qf < 0)
        for (uint32_t i = 0; i < nq; i++)
            if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = (int)i; break; }
    if (qf < 0) return false;
    c.qf = (uint32_t)qf;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = c.qf;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;
    if (vkCreateDevice(c.pd, &di, nullptr, &c.dev) != VK_SUCCESS) { c.dev = VK_NULL_HANDLE; return false; }
    if (!load_device_fns(c.dev)) return false;
    vkGetDeviceQueue(c.dev, c.qf, 0, &c.queue);

    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = c.qf;
    if (vkCreateCommandPool(c.dev, &pi, nullptr, &c.pool) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo ci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ci.commandPool = c.pool;
    ci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ci.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(c.dev, &ci, &c.cmd) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding lb[2] = {};
    for (int i = 0; i < 2; i++) {
        lb[i].binding = (uint32_t)i;
        lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lb[i].descriptorCount = 1;
        lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 2;
    li.pBindings = lb;
    if (vkCreateDescriptorSetLayout(c.dev, &li, nullptr, &c.dsl) != VK_SUCCESS) return false;

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &c.dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(c.dev, &pli, nullptr, &c.pl) != VK_SUCCESS) return false;

    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(SPV);
    smi.pCode = SPV;
    VkShaderModule sm;
    if (vkCreateShaderModule(c.dev, &smi, nullptr, &sm) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm;
    cpi.stage.pName = "main";
    cpi.layout = c.pl;
    VkResult pr = vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &c.pipe);
    vkDestroyShaderModule(c.dev, sm, nullptr);
    if (pr != VK_SUCCESS) { c.pipe = VK_NULL_HANDLE; return false; }

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(c.dev, &dpi, nullptr, &c.dpool) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = c.dpool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &c.dsl;
    if (vkAllocateDescriptorSets(c.dev, &dai, &c.set) != VK_SUCCESS) return false;

    // tiny sink buffer, any memory type
    Buf s;
    uint32_t bits = storage_type_bits(c);
    int t = find_type(c, bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    if (t < 0) t = find_type(c, bits, 0, 0);
    if (t < 0 || !alloc_buf(c, 256, t, s)) return false;
    c.sink = s.buf;
    c.sink_mem = s.mem;
    return true;
}

void teardown(Ctx &c)
{
    if (!c.dev) return;
    vkQueueWaitIdle(c.queue);
    if (c.sink) vkDestroyBuffer(c.dev, c.sink, nullptr);
    if (c.sink_mem) vkFreeMemory(c.dev, c.sink_mem, nullptr);
    if (c.dpool) vkDestroyDescriptorPool(c.dev, c.dpool, nullptr);
    if (c.pipe) vkDestroyPipeline(c.dev, c.pipe, nullptr);
    if (c.pl) vkDestroyPipelineLayout(c.dev, c.pl, nullptr);
    if (c.dsl) vkDestroyDescriptorSetLayout(c.dev, c.dsl, nullptr);
    if (c.pool) vkDestroyCommandPool(c.dev, c.pool, nullptr);
    vkDestroyDevice(c.dev, nullptr);
    c.dev = VK_NULL_HANDLE;
}

const char *type_name(VkPhysicalDeviceType t)
{
    switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
    default: return "GPU";
    }
}

void print_result(const char *label, double gbs, const Buf &b, bool igpu, double ram_peak)
{
    if (gbs < 0) { printf("  %-13s error during measurement\n", label); return; }
    printf("  %-13s measured %7.1f GB/s   (%llu MiB buffer", label, gbs, (unsigned long long)(b.size >> 20));
    if (igpu && ram_peak > 0) printf(", %.0f%% of RAM theoretical", 100.0 * gbs / ram_peak);
    printf(")\n");
}

} // namespace

void vk_run(int samples, double ram_peak)
{
    progress("Vulkan initializing...");
    void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) { progress_clear(); printf("Vulkan  not available (libvulkan.so.1 not found)\n"); return; }

    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    VkInstance inst = VK_NULL_HANDLE;
    if (vkGetInstanceProcAddr)
        vkCreateInstance = (PFN_vkCreateInstance)vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vrambw";
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &app;
    if (!vkCreateInstance || vkCreateInstance(&ii, nullptr, &inst) != VK_SUCCESS) {
        progress_clear();
        printf("Vulkan  not available (no Vulkan driver)\n");
        dlclose(lib);
        return;
    }
#define LI(name) name = (PFN_##name)vkGetInstanceProcAddr(inst, #name);
    LI(vkDestroyInstance) LI(vkEnumeratePhysicalDevices) LI(vkGetPhysicalDeviceProperties)
    LI(vkGetPhysicalDeviceMemoryProperties) LI(vkGetPhysicalDeviceQueueFamilyProperties)
    LI(vkCreateDevice) LI(vkGetDeviceProcAddr)
#undef LI

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    progress_clear();

    int ngpu = 0;
    for (auto pd : pds) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd, &p);
        if (p.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) ngpu++;
    }
    printf("Vulkan  %d GPU(s)%s\n\n", ngpu, n > (uint32_t)ngpu ? " (software devices skipped)" : "");

    int idx = 0;
    for (auto pd : pds) {
        Ctx c;
        c.pd = pd;
        vkGetPhysicalDeviceProperties(pd, &c.props);
        if (c.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;
        vkGetPhysicalDeviceMemoryProperties(pd, &c.mem);
        bool igpu = c.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        uint32_t v = c.props.apiVersion;
        printf("VK %d  %s  %s  Vulkan %u.%u\n", idx++, c.props.deviceName, type_name(c.props.deviceType),
               VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v));
        fflush(stdout);

        progress("  creating Vulkan device...");
        if (!setup(c)) {
            progress_clear();
            printf("  setup failed\n\n");
            teardown(c);
            continue;
        }
        progress_clear();
        uint32_t bits = storage_type_bits(c);

        // device-local memory: VRAM (dGPU) or the memory reserved for the GPU (iGPU).
        // Prefer a type that is not host visible (avoids the small BAR window).
        int dt = find_type(c, bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        if (dt < 0) dt = find_type(c, bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        // host RAM read by the GPU: over PCIe (dGPU), system RAM (iGPU)
        int ht = find_type(c, bits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        Buf b;
        if (dt >= 0 && alloc_test_buf(c, dt, b)) {
            double g = measure(c, b, samples, "device-local");
            print_result("device-local", g, b, igpu, ram_peak);
            free_buf(c, b);
        } else {
            printf("  device-local not available\n");
        }
        if (ht >= 0 && alloc_test_buf(c, ht, b)) {
            double g = measure(c, b, samples, "host RAM");
            print_result("host RAM", g, b, igpu, ram_peak);
            free_buf(c, b);
        } else if (ht < 0) {
            printf("  host RAM      same memory as device-local (unified memory)\n");
        } else {
            printf("  host RAM      allocation failed\n");
        }
        printf("\n");
        teardown(c);
    }
    vkDestroyInstance(inst, nullptr);
    dlclose(lib);
}
