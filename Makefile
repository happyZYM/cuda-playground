CC = nvcc
CFLAGS = -I/usr/local/lib/python3.10/dist-packages/torch/include \
				 -I/usr/local/lib/python3.10/dist-packages/torch/include/torch/csrc/api/include \
				 -lcublas

HELLO = build/hello
HELLO_SRC = csrc/hello_world.cu

BASIC = build/basic
BASIC_SRC = csrc/basic.cu

GEMM = build/gemm
GEMM_SRC = csrc/gemm.cu

all: $(HELLO) $(BASIC) $(GEMM)

$(HELLO): $(HELLO_SRC)
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

$(BASIC): $(BASIC_SRC)
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@ 

$(GEMM): $(GEMM_SRC)
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -rf build

.PHONY: all clean
