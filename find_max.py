problem = "SUMTRIAN"
max = 0

path_to_train = "./CodeChef_Data_ASM_Seq/"+problem+"_Seq_train.txt"
path_to_CV = "./CodeChef_Data_ASM_Seq/"+problem+"_Seq_CV.txt"
path_to_test = "./CodeChef_Data_ASM_Seq/"+problem+"_Seq_test.txt"

max_length = 500
vocab_size = 20000

f = open(path_to_train, "r")
y_train = []
x_train = []
for line in f:
    y, x = line.split("\t\t")
    y_train.append(int(y))
    x_train.append(x)
    if len(x.split()) > max:
        max = len(x.split())


# print(type(x_train))
# print(x_train[0])
# print(x_train.shape) # (number of sentence, max_length)

f = open(path_to_CV, "r")
y_cv = []
x_cv = []
for line in f:
    y, x = line.split("\t\t")
    y_cv.append(int(y))
    x_cv.append(x)
    if len(x.split()) > max:
        max = len(x.split())

f = open(path_to_test, "r")
y_test = []
x_test = []
for line in f:
    y, x = line.split("\t\t")
    y_test.append(int(y))
    x_test.append(x)
    if len(x.split()) > max:
        max = len(x.split())

print(max)