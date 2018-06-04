# Deploying CNN on FPGA using OpenCL  
This is a project for 2017 Innovate FPGA design contest. We hope this project can somehow help those who want to acclerate CNN on resouce-limited embedded systems with FPGA using OpenCL. Origin project link: [PR065](http://www.innovatefpga.com/cgi-bin/innovate/teams.pl?Id=PR065).
## Prerequisites:  
- Board: Terasic [DE10-Nano](http://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&CategoryNo=167&No=1046) with Cyclone V SoC-FPGA (800MHz Dual-core Cortex-A9 processor & 110K LEs FPGA)
- Software: Intel OpenCL SDK 17.1  
## System diagram:  
![System diagram](https://github.com/Er1cZ/Deploying-CNN-on-FPGA-using-OpenCL/blob/master/pic/sys.PNG)
## To use:
- Copy 3 files in `/bin/` folder to `/your_path/` on the TF card for DE10-Nano with Terasic Offical OpenCL BSP image
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

<img src="https://github.com/Er1cZ/Deploying-CNN-on-FPGA-using-OpenCL/blob/master/pic/dog.jpg" width="400px"/> 

Result should be like this: 
```
SqueezeNet on FPGA start:
kernel version 1.2

conv1 takes: 131.289 ms
block1 takes: 505.356 ms
block2 takes: 387.498 ms
block3 takes: 511.104 ms
classifier takes: 495.723 ms
total: 2030.969 ms

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
; Logic utilization                      ;   97%                     ;
; ALUTs                                  ;   65%                     ;
; Dedicated logic registers              ;   39%                     ;
; Memory blocks                          ;   96%                     ;
; DSP blocks                             ;   60%                     ;
+----------------------------------------+---------------------------;  
```
This is a simple and relatively naive implement. We barely found any tutorials and suffered a lot from getting started on this topic. So we make this project more like a getting started tutorial.  

**For more details, please read** [**A getting started tutorial on FPGA implement of CNN using OpenCL**](https://github.com/Er1cZ/Deploying-CNN-on-FPGA-using-OpenCL/blob/master/GettingStartedTutorial.md).
