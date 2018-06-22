# Deploying CNN on FPGA using OpenCL  
This is a project for 2017 Innovate FPGA design contest. We hope this project can somehow help those who want to accelerate CNN on resouce-limited embedded systems with FPGA using OpenCL. Origin project link: [*PR065*](http://www.innovatefpga.com/cgi-bin/innovate/teams.pl?Id=PR065).
## Prerequisites:  
- Board: Terasic [*DE10-Nano*](http://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&CategoryNo=167&No=1046) with Cyclone V SoC-FPGA (800MHz Dual-core Cortex-A9 processor & 110K LEs FPGA)
- Software: Intel FPGA SDK for OpenCL 17.1  
## System diagram:  
![System diagram](https://github.com/Er1cZ/Deploying_CNN_on_FPGA_using_OpenCL/raw/master/pic/sys.PNG)
## To use:
- Copy 2 files in `/bin/v1.3` folder & `/src/common/synset_words.txt` to `/your_path` on the TF card for DE10-Nano with Terasic Offical OpenCL BSP image
- Set up UART connection between DE10-Nano and PC
- Login as root
- Type in commands:
  - `cd ~`
  - `source ./init_opencl.sh`
  - `cd /your_path/`
  - `aocl program /dev/acl0 squeezenet.aocx`
  - `chmod +x squeezenet`
  - `./squeezenet`  

Input image:  

<img src="https://github.com/Er1cZ/Deploying_CNN_on_FPGA_using_OpenCL/raw/master/pic/dog.jpg" width="400px"/> 

Result should be like this: 
```
SqueezeNet on FPGA start:
kernel version 2.0

conv1 takes: 57.173 ms
block1 takes: 84.526 ms
block2 takes: 81.311 ms
block3 takes: 113.345 ms
classifier takes: 115.184 ms
total: 451.539 ms

predicted label: n02106662 German shepherd, German shepherd dog, German police dog, alsatian

done
```  
Resource usage:  
```
+--------------------------------------------------------------------+
; Estimated Resource Usage Summary                                   ;
+----------------------------------------+---------------------------+
; Resource                               + Usage                     ;
+----------------------------------------+---------------------------+
; Logic utilization                      ;   86%                     ;
; ALUTs                                  ;   57%                     ;
; Dedicated logic registers              ;   34%                     ;
; Memory blocks                          ;   68%                     ;
; DSP blocks                             ;   50%                     ;
+----------------------------------------+---------------------------;  
```
We believe Cyclone V FPGA on DE10-Nano board can be a perfect solution to deploy CNN on embedded systems: using its ARM processor as traditional controller and FPGA as a low power & low latency accelerator.
Our simple 120-line OpenCL implement of FPGA accelerator for CNN vividly demonstrates the accessibility and effectiveness of FPGA in high performance computing. Also, we hope our tutorial can help those who want to set foot on this topic and are having a hard time getting started. 

**For more details, please read** [*A getting started tutorial on FPGA implement of CNN using OpenCL*](https://github.com/Er1cZ/Deploying_CNN_on_FPGA_using_OpenCL/blob/master/GettingStartedTutorial.md).
