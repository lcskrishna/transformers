import argparse
import os
import sys

categories = {'BLAS' : ['MmBackward', 'mm', 'matmul', 'bmm', 'BmmBackward', 'addmm'],
              'Eltwise' : ['add', 'add_', 'mul', 'mul_', 'addcmul', 'addcmul_', 'sqrt', 'sqrt_', 'div', 'div_', 'addcdiv_', 'DivBackward0', \
                           'addcdiv', 'AddBackward0' ],
              'Memcpy/Memset' : ['to', 'torch::autograd::CopyBackwards', 'contiguous', 'torch::autograd::AccumulateGrad', 'ViewBackward', 'clone', \
                                 'detach', 'view' ],
              'Softmax/GELU/Droput' : ['SoftmaxBackward', '_softmax_backward_data', 'GeluBackward', 'gelu_backward',\
                                       'dropout', 'FusedDropoutBackward', '_fused_dropout', 'gelu', 'softmax', \
                                       '_softmax', 'log_softmax', 'LogSoftmaxBackward', '_log_softmax'],
              'Embedding' : ['EmbeddingBackward', 'embedding_backward', 'embedding_dense_backward', \
                             'embedding' ],
              'Normalization' : ['norm', 'NativeLayerNormBackward', 'layer_norm', 'native_layer_norm' ],
              'Other' : ['sum', 'item','_local_scalar_dense', 'is_nonzero', 'UnsafeViewBackward', 'reshape', '_masked_scale', \
                         'empty_strided', '_amp_non_finite_check_and_unscale_', 'TBackward', 'is_floating_point', 'TransposeBackward0'],
              'Loss' : ['NllLossBackward', 'nll_loss_backward'],
             }

class KernelSummary():
    def __init__(self, name, count, kernel_time, percentage):
        self.kernel_name = name
        self.kernel_count = count
        self.kernel_time = kernel_time
        self.kernel_percentage = percentage

    def __str__(self):
        return (" Kernel value is name :  " + str(self.kernel_name) + " count : " + str(self.kernel_count) + " time : " + str(self.kernel_time) + "ms percentage: " + \
                        str(self.kernel_percentage) + "%") 

def read_log_file(log_file):
    fs = open(log_file, 'r')
    lines = fs.readlines()
    fs.close()

    return lines

def list_kernel_map(lines):
    total_len = len(lines)
    kernel_map = {}
    for j in range(2,total_len):
        line = lines[j].rstrip()
        split_line = line.split('|')
        kernel_name = split_line[0]
        percentage = float(split_line[1])
        count = float(split_line[4])
        kernel_time = float(split_line[2])/(1000)
        
        kernel_obj = KernelSummary(kernel_name, count, kernel_time, percentage)
        #print (kernel_obj)
        kernel_map[j - 2] = kernel_obj

    return kernel_map 

def list_kernel_summary(kernel_map):
    blas_summary = {}
    category_keys = categories.keys()
    #for i in range(len(categories)):
    for category_key in category_keys:
        values = categories[category_key]
        total_count = 0
        total_percentage = 0
        total_time = 0 
        for j in range(len(kernel_map)):
            kernel_obj = kernel_map[j]
            name = kernel_obj.kernel_name
            if name in values:
                total_count = total_count + kernel_obj.kernel_count
                total_percentage = total_percentage + kernel_obj.kernel_percentage
                total_time = total_time + kernel_obj.kernel_time
        print ("Category Key : {}, Total_count : {}, Total_Percentage : {}, Total_Time : {}".format(category_key, total_count, total_percentage, total_time/1000)) 

def main():
    log_file = os.path.abspath(args.csv_log)
    output_filename = os.path.abspath(args.output_filename)

    lines = read_log_file(log_file)
    kernel_map = list_kernel_map(lines)
    list_kernel_summary(kernel_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-log", type=str, required=True, help="CSV File generated from autograd")
    parser.add_argument("--output-filename", type=str, required=False, default="kernel_summary.csv", help="Outputfilename containing kernel summary")

    args = parser.parse_args()

    main()
