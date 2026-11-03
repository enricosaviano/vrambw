-include config.mk

TARGET = vrambw
OBJ    = src/main.o src/ram.o

ifeq ($(HAVE_CUDA),1)
OBJ    += src/gpu.o
DEFS    = -DHAVE_CUDA
LINK    = $(NVCC) $(CCBIN)
else
LINK    = $(CXX) -pthread
endif

ifeq ($(wildcard config.mk),)
all:
	@echo "config.mk missing: run ./configure first"; exit 1
else
all: $(TARGET)
endif

$(TARGET): $(OBJ)
	$(LINK) -o $@ $(OBJ) -lpthread

src/%.o: src/%.cpp src/ram.h src/gpu.h src/progress.h config.mk
	$(CXX) $(CXXFLAGS) $(DEFS) -c -o $@ $<

src/gpu.o: src/gpu.cu src/gpu.h src/progress.h config.mk
	$(NVCC) $(NVFLAGS) $(CCBIN) $(GENCODE) -c -o $@ $<

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET) src/*.o

distclean: clean
	rm -f config.mk

.PHONY: all run clean distclean
