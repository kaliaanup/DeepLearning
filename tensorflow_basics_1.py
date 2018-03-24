import tensorflow as tf
import numpy as np
import matplotlib.image as mp_image
import matplotlib.pyplot as plt

# The ~same simple calculation in Tensorflow
x = tf.constant(1, name='x')
y = tf.Variable(x+10, name='y')
print y
print x
#When the variable y is computed, take the value of the constant x and add 10 to it


#To actually calculate the value of the y variable and to evaluate expressions, 
#we need to initialise the variables, and then create a session where the actual 
#computation happens

model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    print(session.run(y))
    

a = tf.constant(5, name="a")
b = tf.constant(45, name="b")
y = tf.Variable(a+b*2, name='y')
model = tf.global_variables_initializer()

with tf.Session() as session:
    # Merge all the summaries collected in the default graph.
    merged = tf.summary.merge_all() 
    
    # Then we create SummaryWriter. 
    # It will write all the summaries (in this case the execution graph)
    #obtained from the code's execution into the specified path
    writer = tf.summary.FileWriter("tmp/tf_logs_simple", session.graph)
    session.run(model)
    print(session.run(y))

#variable
my_variable = tf.get_variable("my_variable", [1, 2, 3])
my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32,
  initializer=tf.zeros_initializer)

#variable collections
my_local = tf.get_variable("my_local", shape=(),
collections=[tf.GraphKeys.LOCAL_VARIABLES])
tf.add_to_collection("my_collection_name", my_local)



#one dimensional tensor
tensor_1d = np.array([1, 2.5, 4.6, 5.75, 9.7])
tf_tensor=tf.convert_to_tensor(tensor_1d,dtype=tf.float64)
with tf.Session() as sess: 
    print(sess.run(tf_tensor))
    print(sess.run(tf_tensor[0]))
    print(sess.run(tf_tensor[2]))

#two dimensional tensor
tensor_2d = np.arange(16).reshape(4, 4)
print(tensor_2d)
tf_tensor = tf.placeholder(tf.float32, shape=(4, 4))
with tf.Session() as sess:
    print(sess.run(tf_tensor, feed_dict={tf_tensor: tensor_2d}))

#basic matrix operations
matrix1 = np.array([(2,2,2),(2,2,2),(2,2,2)],dtype='float32') 
matrix2 = np.array([(1,1,1),(1,1,1),(1,1,1)],dtype='float32')

tf_mat1 = tf.constant(matrix1) 
tf_mat2 = tf.constant(matrix2)

matrix_product = tf.matmul(tf_mat1, tf_mat2)
matrix_sum = tf.add(tf_mat1, tf_mat2)

matrix_det = tf.matrix_determinant(matrix2)

with tf.Session() as sess: 
    prod_res = sess.run(matrix_product) 
    sum_res = sess.run(matrix_sum) 
    det_res = sess.run(matrix_det)
    
print("matrix1*matrix2 : \n", prod_res)
print("matrix1+matrix2 : \n", sum_res)
print("det(matrix2) : \n", det_res) #described as the scaling factor determinant of mat A is |A|

filename = "imgs/keras-logo-small.jpg"
input_image = mp_image.imread(filename)

#dimension
print('input dim = {}'.format(input_image.ndim))
#shape
print('input shape = {}'.format(input_image.shape)) #(300,300,3) ht, width, color

#plt.imshow(input_image)
#plt.show()

#slicing

my_image = tf.placeholder("uint8",[None,None,3])
slice = tf.slice(my_image,[0,0,0],[16,-1,-1])

with tf.Session() as session:
    result = session.run(slice,feed_dict={my_image: input_image})
    print(result.shape)
    
#plt.imshow(result)
#plt.show()    

#transpose
x = tf.Variable(input_image,name='x')
model = tf.global_variables_initializer()

with tf.Session() as session:
    x = tf.transpose(x, perm=[1,0,2])
    session.run(model)
    result=session.run(x)
    
# plt.imshow(result)
# plt.show()


#compute gradients
x = tf.placeholder(tf.float32)
y = tf.log(x)  
var_grad = tf.gradients(y, x)
with tf.Session() as session:
    var_grad_val = session.run(var_grad, feed_dict={x:2})
    print(var_grad_val)
    