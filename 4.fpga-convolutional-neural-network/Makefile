# Makefile for running Vitis HLS scripts

# Default target
all: vitis csim

# Run synthesis script
vitis:
	vitis_hls -f vitis.tcl

# Run C simulation script
csim:
	#vitis_hls -f csim.tcl
	# Please run ulimit -s unlimited
	# You can try vitis_hls -f csim.tcl but it is faster to run the csim.cpp directly
	g++ csim.cpp cnn.cpp -I/tools/Xilinx/Vitis_HLS/2023.2/include/ -O3 -o csim.out
	./csim.out

# Clean intermediate HLS project directories (optional)
clean:
	rm -rf *.log cnn.prj

zip:
	sudo apt install -y zip
	cp cnn.prj/solution/syn/report/kernel_cnn_csynth.rpt .
	zip -r lab4.zip kernel_cnn_csynth.rpt lab4-report.pdf cnn.cpp

.PHONY: all vitis csim clean