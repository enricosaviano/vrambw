-include config.mk

TARGET = vrambw
OBJ    = src/main.o src/ram.o src/meminfo.o
DEFS   =
LIBS   = -lpthread

ifeq ($(HAVE_CUDA),1)
OBJ    += src/gpu.o
DEFS   += -DHAVE_CUDA
LINK    = $(NVCC) $(CCBIN)
else
LINK    = $(CXX) -pthread
endif

ifeq ($(HAVE_VULKAN),1)
OBJ    += src/vk.o
DEFS   += -DHAVE_VULKAN
LIBS   += -ldl
endif

ifeq ($(HAVE_ROCM),1)
OBJ    += src/hip.o
DEFS   += -DHAVE_ROCM
ifeq ($(HAVE_CUDA),1)
LIBS   += -L$(ROCM_DIR)/lib -lamdhip64 -Xlinker -rpath -Xlinker $(ROCM_DIR)/lib
else
LIBS   += -L$(ROCM_DIR)/lib -lamdhip64 -Wl,-rpath,$(ROCM_DIR)/lib
endif
endif

HDRS = src/ram.h src/gpu.h src/hip.h src/meminfo.h src/vk.h src/progress.h

ifeq ($(wildcard config.mk),)
all:
	@echo "config.mk missing: run ./configure first"; exit 1
else
all: $(TARGET)
endif

$(TARGET): $(OBJ)
	$(LINK) -o $@ $(OBJ) $(LIBS)

src/%.o: src/%.cpp $(HDRS) config.mk
	$(CXX) $(CXXFLAGS) $(VK_CFLAGS) $(DEFS) -c -o $@ $<

src/vk.o: src/vk_read.spv.inc

src/hip.o: src/hip.cpp src/hip.h src/progress.h config.mk
	$(HIPCC) -O3 -std=c++14 -fPIC $(HIP_ARCH) -c -o $@ $<

src/gpu.o: src/gpu.cu src/gpu.h src/progress.h config.mk
	$(NVCC) $(NVFLAGS) $(CCBIN) $(GENCODE) -c -o $@ $<

# regenerate the embedded Vulkan shader (only needed after editing vk_read.comp; needs glslc)
shader:
	glslc -O --target-env=vulkan1.0 -mfmt=c -o src/vk_read.spv.inc src/vk_read.comp

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET) src/*.o

distclean: clean
	rm -f config.mk

.PHONY: all run clean distclean shader
