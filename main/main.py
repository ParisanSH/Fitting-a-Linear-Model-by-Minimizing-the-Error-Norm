import random
import numpy as np
import matplotlib.pyplot as plt

# Cost_func for L2 --> MSE=1/N ∑ (i=1, n)(yi−(wxi+b))**2 -------------------------
def cost_function(x__, y_hat__, w_current, b_current):
    n = len(x__)
    total_error=0.0
    for i in range(n):
        total_error += (y_hat__[i]-(w_current*x__[i] + b_current))**2
    return total_error/n


# Gradient descent to calculate the gradient of our cost function (in L2) ----------
def update_w(x__, y_hat__ , w_current , b_current ,learning_rate ):
    w_deriv = 0
    b_deriv = 0
    n = len(x__)

    for i in range(n):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        w_deriv += -2*x__[i] * (y_hat__[i] - (w_current* x__[i] + b_current))

        # -2(y - (mx + b))
        b_deriv = -2*(y_hat__[i] - (w_current*x__[i] + b_current))
    
    w_current -= (w_deriv / float(n)) * learning_rate
    b_current -= (b_deriv / float(n)) * learning_rate 

    return w_current , b_current

# Training a model (norm_2) ---------------------------------------------------------
# Training is complete when we reach an acceptable error threshold, 
# or when subsequent training iterations fail to reduce our cost.
def train_l2( x_ , y_hat_ , w_ ,b_ , learning_rate , iters):
    
    cost_func_history = []
    for i in range(iters):
        w_ , b_ = update_w(x_ , y_hat_ , w_ , b_ , learning_rate)

        #Calculate cost for auditing purposes
        cost = cost_function(x_ , y_hat_ , w_ , b_)
        cost_func_history.append(cost)
        if i % 10 == 0:
            print ("iter: "+str(i) + " cost: "+str(cost))
            y = x_ * w_ + b_
            #plt.plot(x_ , y ,ls = '--')
            #plt.show()

    return w_ , b_ , cost_func_history

# Plot for L2 -----------------------------------------------------------------------
def show_plot_l2(x , y_hat ,W , B , n_iterations , cost_history):
    plt.scatter(x ,y_hat, c='purple') # show data point 

    y_hat = x*W + B 

    plt.plot(x , y_hat , c= 'red', ls = '--')
    plt.xlabel('x axis ')
    plt.ylabel('y_hat axis')
    plt.grid()
    plt.title('Norm - L2 ')
    plt.show()

    plt.plot(np.arange(n_iterations) , cost_history)
    plt.grid()
    plt.xlabel('Training Iterations ')
    plt.ylabel('Mean Squared Error ')
    plt.title('Error rate')
    plt.show()

# L1 functions ----------------------------------------------------------------------
def cost_func_l1(w_ , b_ , x_ , y_):
    y_tilda = w_ * x_ + b_
    error = np.sum(np.abs(y_tilda - y_))
    return error , y_tilda

def update_W_B(w_l , b_l , w_u , b_u , x__ ,y__):
    
    #np.sum(np.abs(y_tilda_1 - y_))
    error_u , y_u = cost_func_l1(w_u,b_u ,x__ , y__)
    error_l , y_l= cost_func_l1(w_l,b_l ,x__ , y__)
    if error_u < error_l:
        b_l = (b_l + b_u)/2
        w_l = (w_l + w_u)/2
        #w_l += .5
    else:
        b_u = (b_l + b_u)/2
        w_u = (w_l + w_u)/2
        #w_u -= .5

    return w_l , b_l , w_u , b_u

# Trai_l1 ---------------------------------------------
def train_l1 ( x_ , y_ , iters ):
    cost_history = []

    wL = -100
    wU = 100
    bL = random.uniform(-100,100)
    bU = random.uniform(-100,100)
    
    for i in range(0,iters):
        wL , bL , wU , bU = update_W_B(wL , bL , wU , bU , x_ ,y_)
        error , yU = cost_func_l1(wU , bU , x_ , y_)
        cost_history.append(error)
        if i % 10 == 0:
            print ("iter: "+str(i) + " cost: "+str(error))
            #plt.plot(x_ , yU ,ls = '--')
            
    errorU , yU = cost_func_l1(wU , bU , x_ , y_)
    errorl , yl = cost_func_l1(wL , bL , x_ , y_)

    if errorl <= errorU :
        return wL , bL , cost_history
    else:
        return wU , bU , cost_history

# Linfi functions -------------------------------------------------------------------

def cost_func_linfi(w_ , b_ , x_ , y_):
    y_tilda = w_ * x_ + b_
    error = np.max(np.abs(y_tilda - y_))
    return error , y_tilda

def update_W_B_infi(w_l , b_l , w_u , b_u , x__ ,y__):
    
    #np.sum(np.abs(y_tilda_1 - y_))
    error_u , y_u = cost_func_linfi(w_u,b_u ,x__ , y__)
    error_l , y_l= cost_func_linfi(w_l,b_l ,x__ , y__)
    if error_u < error_l:
        b_l = (b_l + b_u)/2
        w_l = (w_l + w_u)/2
        #w_l += .5
    else:
        b_u = (b_l + b_u)/2
        w_u = (w_l + w_u)/2
        #w_u -= .5

    return w_l , b_l , w_u , b_u

# Trai_linfi ---------------------------------------------
def train_linfi ( x_ , y_ , iters ):
    cost_history = []

    wL = -100
    wU = 100
    bL = random.uniform(-100,100)
    bU = random.uniform(-100,100)
    
    for i in range(0,iters):
        wL , bL , wU , bU = update_W_B_infi(wL , bL , wU , bU , x_ ,y_)
        error , yU = cost_func_linfi(wU , bU , x_ , y_)
        cost_history.append(error)
        if i % 10 == 0:
            print ("iter: "+str(i) + " cost: "+str(error))
            #plt.plot(x_ , yU ,ls = '--')
            
    errorU , yU = cost_func_linfi(wU , bU , x_ , y_)
    errorl , yl = cost_func_linfi(wL , bL , x_ , y_)

    if errorl <= errorU :
        return wL , bL , cost_history
    else:
        return wU , bU , cost_history

# L1 functions ----------------------------------------------------------------------

def cost_func_l0(m_ , b_, x , y):
    n = len(x)
    cost = 0
    y_tilda = m_*x + b_
    y_hat = y_tilda - y
    for i in range(n):
        if y_hat[i] != 0:
            cost += 1
    return cost

def train_l0 ( x , y ,n):
    
    m = []
    b = []
    cost_ = []
    count = 0
    for i in range(n):
        for j in range (i+1 , n):
            m.append((y[j]-y[i])/(x[j]- x[i]))
            b.append(-m[count]*x[i]+ y[i])
            count += 1
    for i in range (count):
        #m , b = update_m_b_l0 ( m , b , x , y)
        cost = cost_func_l0(m[i] , b[i], x , y)
        cost_.append(cost)
    cost = min(cost_)
    #print(count)
    M = sum(m)/ len(m)
    #print(M)
    index = 0
    for i in range(count):
        if np.abs(M - m[i]) < 0.01 and cost_[i] == cost:
            M = m[i]
            index = i 
            break
    w_answer = M
    b_answer = b[index]
    return w_answer , b_answer
# Main  -----------------------------------------------------------------------------

# set x and y_hat 
n = int(input('enter n:'))
x = np.random.uniform(1.0,200.0,n)
#print (x)

W = random.uniform(-100,100)
B = random.uniform(-100,100)

Y = x*W + B
plt.plot(x , Y , c = 'blue')
#print(Y)
#y_hat = Y +  np.random.uniform(0,10.0,n)
y_hat = np.zeros(n, float)
for i in range(n):
    y_hat[i]= Y[i]+ random.uniform(0,10)
#print(y_hat)   

# L2 -------------------------------------------------------------------------------- 
n_iterations = 500
learning_rate = 0.00001
w = random.uniform(-10,10)
b = random.uniform(-10,10)
cost_history = []
print(W ,B )
W , B , cost_history = train_l2 ( x , y_hat , w , b ,learning_rate , n_iterations )
print(W ,B )
# draw plot for L2
show_plot_l2(x , y_hat ,W , B , n_iterations , cost_history)

# L1 --------------------------------------------------------------------------------
cost_his = []
w , b , cost_his = train_l1 ( x , y_hat , n_iterations )
print( w , b)
y_tilda = x*w+ b
plt.plot(x , Y , c = 'blue')
plt.scatter(x ,y_hat, c='purple') # show data point
plt.plot(x , y_tilda ,ls = '--', c = 'green')
plt.xlabel('x axis ')
plt.ylabel('y_hat axis')
plt.grid()
plt.title('Norm - L1 ')
plt.show()

plt.plot(np.arange(n_iterations) , cost_his)
plt.grid()
plt.xlabel('Training Iterations ')
plt.ylabel('Mean Squared Error ')
plt.title('Error rate')
plt.show()

# Linfini --------------------------------------------------------------------------------

cost_his = []
w , b , cost_his = train_linfi ( x , y_hat , n_iterations )
print("w' = %f , b' = %f" %(w , b))
y_tilda = x*w+ b

plt.plot(x , Y , c = 'blue')
plt.scatter(x ,y_hat, c='purple') # show data point
plt.plot(x , y_tilda ,ls = '--', c = 'green')
plt.xlabel('x axis ')
plt.ylabel('y_hat axis')
plt.grid()
plt.title('Norm - Linfinity ')
plt.show()

plt.plot(np.arange(n_iterations) , cost_his)
plt.grid()
plt.xlabel('Training Iterations ')
plt.ylabel('Mean Squared Error ')
plt.title('Error rate')
plt.show()

# L0 --------------------------------------------------------------------------------


w_ans , b_ans = train_l0 ( x , y_hat ,n)

y_ans = x * w_ans + b_ans
print("w' = %f , b' = %f" %(w_ans, b_ans))

plt.plot(x , Y , c = 'blue')
plt.scatter(x ,y_hat, c='purple') # show data point
plt.plot(x , y_ans ,ls = '--', c = 'green')
plt.xlabel('x axis ')
plt.ylabel('y_hat axis')
plt.grid()
plt.title('Norm - L0 ')
plt.show()