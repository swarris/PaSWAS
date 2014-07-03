
# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../gpuAlign.cu \
../smithwaterman.cu 

CU_DEPS += \
./gpuAlign.d \
./smithwaterman.d 

OBJS += \
./gpuAlign.o \
./smithwaterman.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	nvcc --ptxas-options=-v -keep -arch=compute_30 -I/usr/local/cuda-6.0/samples/common/inc/ -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


