# NG
## Main Topics
* Supervised Learning (x(i), y(i)) 
	* Linear regression, logistic regression, neural networks, SVMs
* Unsupervised Learning x(i)
	* K-means, PCA, Anomaly detection
* Applications
	* Recommender systems, large scale machine learning
* Tuning /debugging
	* Bias/variance, regularization
	* Learning curves, error analysis, ceiling analysis

### Class Terms
* Assume 1-indexed vectors
* Capitals for matrices, lowercase for matrices, scalars, numbers
* m - # training examples (rows)
* n - number of features
* x’s - input variable / features
* y’s - output variable /target
* (x, y) for one training example
* x^i, y^i denotes ith index
* x_j^i value of feature /j/ in /ith/ training example
* J - loss function that outputs error
* theta - coefficients
* h - hypothesis, a function that maps from x’s to y’s
	* To describe the supervised learning problem slightly more formally, our goal is,
	* given a training set, to learn a function h : X → Y so that h(x) is a “good” predictor for the corresponding value of y. For historical reasons, this function h is called a hypothesis.
* a := 1 is assignment
	* a = a+1 is assertion check
* Contour graph is like looking at one of those 3d graphs from the top.  A similarly colored line would yield the same loss.

## Regression
### Gradient Descent
* Gradient descent takes steps, evaluating and looking for the minimum.  However, it is possible that it will get stuck in a local minimum, and not find the true global minimum
* The alpha / learning rate decides how large the adjustments are in each step of gd
* The correct implementation of GD attempts to simultaneously update all coefficients (theta0, theta1….thetaN)
	* So calculate for all, then do assignments. Don’t minimize theta0, assign, then perform minimization for theta1
	
### Matrices and Vertices
* 4x2 is 4 rows, 2 columns
* A_12 (subscript of 12) refers to row 1 column 2
* Vector
	* Special matrix that has one column (n x 1)
* 4x1 can be called a four dimensional vector
* Can only add matrices that are the same size
* For matrix-vector multiplication, the first multiple’s columns has to match the second’s rows  (3x2 and 2x1 is valid, and results in 3x1 matrix) 
	* Result is a vector where number of rows is equal to number of rows in first multiple
	* The result is a *vector*. The number of *columns* of the matrix must equal the number of *rows* of the vector.
	* An *m x n matrix* multiplied by an *n x 1 vector* results in an *m x 1 vector*.
* Solving the loss function in one shot without calculating all iteratively involves matrices
* Matrix-matrix multiplication
	* The number of columns in the first operand must match the number of rows in the second
	* (m by n) x (n by o) = matrix of size (m by o)
	* Not commutative (AxB /= BxA)
* Identity matrix
	* np.identity(size)
	* For matrix A and identity matrix I, `AxI = IxA = A`
	
### Feature Scaling
* Feature scaling helps convergence happen more quickly
* Scale to -1 <= x <= 1
	* 0 <= x <= 3, -2 <= x <= 0.5 are fine,  -3 to 3 okay too!
	* -100 <= x < 2000 is not
	* A very small range is not good either (0.00001 difference)
	* On a contour graph, one feature with a large range and one with a small one 
* Mean Normalization
	* Apply transformation so that mean of features is approximately 0
		* One way to do so being `(x - u) / s`
			* x - the current feature, u - the average of all features, s - the range of the features OR the standard deviation of the features
			
### Gradient Descent
* If error is increasing/oscillating, try decreasing learning rate. It may be that gradient descent is always overshooting minimum
* A sufficiently small alpha should always lead to convergence, but will increase time to convergence
	* Try 0.001, 0.003, 0.01, 0.03, 1, ….
	
### Normal Equation
* theta = (X^T *X)^-1 * X^T * y
* (X^T *X)^-1 being the inverse of X^TxX
* octave: `pinv(X’*X)*X'*y`
	* gives the optimal v
	* alue of theta that minimizes the cost function
	* side note: `pinv` is pseudo inverse and will still go through even if X^T*X is non-invertible
* Non-invertible X^T *X occurs with
	* redundant features (collinearity and otherwise)
	* too many features
		* delete some features or use regularization
* *GD vs Normal*
	* /m/ training examples, /n/ features
	* Normal advantages
		* GD needs to chose learning rate, needs many iterations
		* N doesn’t need learning rate, doesn’t need to iterate
	* GD advantages
		* Works well even when n is large
		* N needs to compute (X^T * X)^-1
			* Therefore slow with many features
			* Cost of O(n^3)
	* Normal better than GD usually up to 10k features
	* Normal doesn’t work for the more complicated algorithms

### Logistic Regression
* Want  0 <= h(x) <= 1
	* `h(x) = P( y = 1 | x; theta )`
		* probability that y=1 given x, parameterized by theta
* h(x) = g(theta^T * x), the sigmoid / logistic function
	* Where g(z) = 1 / ( 1 + e^-z)
	* x = [ 1; tumorSize ], h(x) =0.7 => y=1 by threshold
* g(z) only outputs between 0 and 1, which is what we want
*  `P(y=0|x; theta) + P(y=0|x; theta)  = 1`
	* chance of y being 0 and chance of y being 1 adds up to 1
	* Therefore `P(y=0) = 1 - P(y=1)`
* Decision Boundary
	* Suppose predict y=1 if h(x) >= 0.5, else y=0
	* In the sigmoid function g(z), g(z) >= 0 when z > 0
		* i.e. `theta^T*x > 0` means `y = 1`
	* Once you have the parameters theta, then the decision boundary is defined
	* h(x) of theta0 + theta1x, where t0 is 0 and t1 is 1,
		* then y =1 if x > 0
* Choosing Theta, Optimization Objective / Cost Function
	* The original cost function would result in a non-convex loss, so we need to use a different one to ensure that a global minimum is found
	* Cost(h(x), y) = 
		* `-log(h(x))` if y =1 
			* concave facing right graph that appropriately lowers cost as h(x) approaches y. asymptote at 0, so as h(x) approaches 0, coat approaches infinity
		* `-log(1 -h(x))` if y=0 
			* concave facing left graph with asymptote at 1, approaches infinity as h(x) approaches 1.  this applies the appropriate penalty
		* if h(x) = y, then cost(h(x), y) = 0 for y=0 and y=1
		* if y = 0, then cost(h(x), y) approaches inf. as h(x) approaches 1
		* if h(x) = 0.5, then cost(h(x), y) > 0 regardless if y=0 or 1
	* Thus the cost function can be compressed to:
		* ` cost(h(x), y) = -y * log(h(x)) - (1 - y ) * log( 1 - h(x) ) `
			* If y =1, - y being 1 and (1-y) being 0, second term is multiplied out
				* `If y = 1, Cost(h(x), y) = -log(h(x))`
			* ` If y = 0, Cost(h(x) = -log( 1 - h(x) )` 
* Final Cost Function
	* -(1/m) * SUM(from 1 to m) of cost(h(x),y) = -y*log(h(x)) - (1 -y ) * log( 1 - h(x) )
	* Vectorized:
		* h = g(X * theta)
		* J(theta) = (1/m) * (-y^T * log(h) - (1 - y)^T * log( 1 - h))  
*  Gradient Descent
	* Taking the derivative of the gradient term…
	* theta = theta - alpha * sum(1 to m)( h(x) - y ) * x_j
		* So the general composition is the same as linreg, but h(x) is different 
* Alternative optimization algorithms
	* Conjugate gradient, BFGS, L-BFGS
	* Advantages: No need to pick learning rate, faster than GD
	* Disadvantages: More complex
	
### Multiclass Classification
* One way to do more than 2 classes is to do ’one vs all’ or ‘one vs rest’
* Instead of predicting each class ‘at the same time’, predict only one class and separate all others as the opposite
* Therefore for k classes, requires k classifiers

### Overfitting
* For points on a slight curve, i.e. `x + x^2`
	* A straight line is ‘underfit’ or has ‘high bias’ because it assumes all points will fit on a linear prediction
	* If the prediction fits too hardly to the data points, it will struggle to predict from outside the training set.  Referred to as ‘overfit’ or ‘high variance’
		* ‘generalization’, a term which applies to how well a hypothesis can fit to new examples
* Reduce overfitting by
	* Reducing number of features
		* Manually select which features to keep
		* Model selection algorithm
	* Regularization
		* Keep all features, but reduce magnitude/values of parameters theta_j
		* Works well when we have a lot of features which contribute to predicting y 
		
### Regularization
* Small values for parameters theta_…j
	* ‘simpler’ hypothesis which these small values
	* less prone to overfitting
* If a hypothesis involved 100’s of parameters, can’t really manually go through each to pick, so regularization will help make those decisions for us
* J(t) = (1/2m) * [sum(1 to m) (h(x) - y)^2 + LAMBDA * sum(1 to n)(t ^2) ]
	* Lambda being an expression of ‘trade off’ between two different goals
	* The first goal, captured by `h(x) - y)^2` is that the hypothesis should be fit to the data. 
	* The second goal is to keep the parameters small, captured by the lambda * sum(theta ** 2), i.e. the regularization objective
		* So lambda controls the trade off between the goal of fitting the training set we’ll and and the goal of keeping the parameters small/keeping the hypothesis simple to avoid overfitting 
* Regularization enables the use of a higher order polynomial, and instead of overfitting a very curvy graph, it smooths out the hypothesis to create a much simpler curve to generalize
	* If lambda is set too large, then all the theta parameters will be too small, essentially creating a horizontal line hypothesis, i.e.  too high bias/preconception
	* There are ways of calculating lambda as well. 
	
### Regularized Gradient Descent
* t = theta
* t = t * (1 - alpha* (lambda / m) - alpha * (1/m) * sum(1 to m)(h(x) - y)*x
	* The term `1 - alpha * (lambda / m)` is less than one to produce the desired result
### Regularized Normal Equation
theta = ( X’X + lambda * zeroes(n+1) )^-1 * x’ * y
* Applied to the logistic regression cost function…
  -(1/m) * SUM(from 1 to m) of cost(h(x),y) = -y*log(h(x)) - (1 -y ) * log( 1 - h(x) ) 
  +lambda * theta_j

## Neural Networks
### Neural Networks - Forward Feed
* Terminology:
	* Layer 1 defined as the input layer,  Layer 2 defined as the first hidden layer and so forth
	* `a^j_i` - ‘activation’ of layer j, unit i
	* ` theta^j` - matrix of weights controlling function mapping from layer j to layer j + i
	* s_j is number of activation nodes in layer j
* If a network has s_j units in layer j, s_(j+1) units in layer j+1, then theta will be of dimension s_(j+1) x (s_j + 1)
	*  The theta of a model with 3 inputs and one hidden layer with 3 neurons will be represented with a *3x4 matrix*
	* A model with 2 input neurons and one hidden layer with 4 neurons will have a theta dimension of *4x3*
* z is a variable that encompasses the parameters inside the g function
	* i.e. a = g(theta^1_1 + theta^2_1…) is a = g(z)
* a^j = g(z^j)  
* h(x)  = a^(j+1) = g(z^(j+1))
	* All this boils down to is that between layer j and j+1, the same process as logistic regression is happening
	* Adding these intermediate layers enables the production of more interesting and complex non-linear hypotheses
	
### Simple example: AND
* y = x1 AND x2, both binaries (0 or 1)
* 2 inputs + 1 bias , one output layer to h(x)
	* Diagram *3 -> 1 ->h(x)*
	* Bias unit has weight of -30, so subtracts 30
		* Notation: THETA^1_10
	* Input x1 has weight of +20, so multiplies x1 by 20
		* THETA^1_11
	* Input x2 has weight of +30, so multiplies x2 by 20
		* THETA^1_12
	* Express as  h(x) = g(-30 + 20x1 + 20x2)
		* When plugging these values in, and then into a sigmoid function with a boundary value of 0.5, then will resolve to 1 only when x1 and x2 are 1
* OR
	* h(x) = g(-20 + 10x1 + 10x2)
* (NOT x1) AND (NOT x2)
	* a sufficiently negative theta so that it only produces y=1 when x1=x2=0
	* h(x) = 10 -20x1 - 20x2
* x1 XNOR x2
	* y=1 when x1 and x2 are of the same value
	* on a two-axis graph of x1 and x2, y=1 for upper-right and lower-left, =0 for upper left and lower right.  So this requires a non-linear separation
	* Representation of 3 -> 2 -> 1 -> output
		* 2 inputs, 1 bias (3)
		* These feed to the next layer of two nodes, one is of `x1 AND x2` the other `(NOT x1) AND (NOT x2)` 
		* Now that the 2nd layer has the computed&transformed values, these feed to the last layer which has an activation of `x1 OR x2`.    It uses the values of the second layer’s nodes and applies the OR weights to create the XNOR NN
		
### Multiclass
* Instead of representing y as an integer, for four classes: class1 as [1; 0; 0; 0], class2 as [0; 1; 0; 0], and so forth

### Neural Network Classification
* L = total number of units 
* s_l  = # of units (not counting bias unit) in layer l
* K = output units
	* Binary classification has 1 output unit
		* s_l_L (last layer) = 1, K=1
	* Multi-class classification
	* s_L = K, 
* Gradient computation requires J(theta), and all of the partial derivative terms involved in summing up the result. 

### Backpropagation Gradient Calculation
* delta is partial derivative
* For each node, compute delta^l_j  that represents the error of node j in layer l
* For each output unit ( layer L = 4 ) 
	* delta^4_j = a^4_j - y_j
		* a_j being h(x)_j
		* Therefore this term is just the hypothesis value minus y in the training set
	* Vectorized
		* d^4 = a^4 - y
		* *Each of these is a vector whose dimension is equal to the number of output units in the network*
* delta3 = (theta3 transposed)*delta4  .* g’(z^3)
	* g’(z^3) is the partial derivatives of the activation function evaluated with the input values given by z^3.
		* In vector terms that is a^3 .* (1 - a^3)
			* 1 being a vector of ones, and a being the vector of activation values for that layer
* delta2 = (theta2 transposed)*delta3  .* g’(z^2)
	* Note that the deltas are highlighted because of the dot, the dot to make the multiplication element-wise
* There is no delta^1 because that is the input layer, and therefore has no associated error in that point of the neural network 
* The name back propagation comes from the fact that we start by computing the delta term for the output layer and then we go back a layer and compute the delta terms for the third hidden layer, and then back another step to compute delta^2.  
	* Doing so ‘back propagates’ the errors from the output layer to the previous layer and so on. 
* Suppose two training examples (x1, y1) and (x2, y2).  The correct sequence of operation to compute the gradient is:
	* FP using x1, followed by BP using y1.  Then FP using x2 followed by BP using y2.
* cost(i) is similar to squared error (h(x) -y)^2, how close is the hypothesis is to the prediction
### Backpropagation intuition
* simple binary classification w/o regularization can be expressed as
` cost(t) = y^t * log(h(x^t)) + (1-y^t) * log(1 - h(x^t))`
* Delta^l_j is the ‘error’ for a^l_j, and is the derivative of the cost function
* The derivative is the slope of a line tangent to the cost function, so the steeper the slop the more incorrect the hypothesis is.  
### Unrolling matrices into vectors
* Have initial parameters Theta1, Theta2, Theta3
* Unroll to get initialTheta to pass to `fminunc(@costFunction, initialTheta, options)`
* cost function `[jval, gradientVec] = costFunction(thetaVec)`
	* From thetaVec, get Theta123 (reshape to get back original thetas)
	* Use forward /back prop to compute D123 and J(theta)
	* Unroll D123 to get gradientVec which is now what the cost function can return
* Advantage of being in matrices is that it’s more convenient to do forward and back propagation and easier 
* Advantage of being in a vector is in using the advanced optimization algorithms, which assume you have unrolled all your parameters into a big long vector

### Backpropagation with Numerical Gradient Checking
* Many ways to have subtle bugs in back propagation
	* So that if run with gradient descent or another optimization algorithm, it may look as if it’s working with cost/J(theta) decreasing on every iteration of gradient descent. 
* Gradient checking is an idea that eliminates these bugs, and ensures a high level of confidence with forward and backwards propagations
* Implementation
	* Implement backprop to compute DVec (unrolled D1, D2, D3)
	* Implement numerical gradient check to compute gradApprox
	* Make sure they have similar values
	* Turn off gradient checking, using backdrop code for learning
* Important
	* Be sure to disable gradient checking code before training classifier. If you run numerical gradient computation for every iteration of gradient descent, the code will be very slow.
	
### Random Initialization
* For gradient descent and advanced optimization methods, need initial value for Theta
	* `optTheta = fminunc(@costFunction, initialTheta, options)`
* Consider gradient descent - set initialTheta = zeros(n,1)?
	* Although this is good for logistic regression, it does not work for neural networks
	* This would mean that the value of a from one layer to another would not change, and the partial derivatives would result in the same weights (symmetry)
* Random initialization: Symmetry breaking
	* Initialize each Theta to a random value in [-Epsilon, Epsilon
	
## Architecture
### Putting it together
* Choosing an Architecture
	* No of input units: Dimension of features x^i
	* No. output units: Number of classes
	* Reasonable default: 1 hidden layer, or if >1 hidden layer, have same under of hidden units in every layer ( usually the more the better)
* Training a Neural Network
	* Randomly initialize weights
	* Implement forward propagation to get h(x) for any x
	* Implement code to compute cost function J(theta)
	* Implement backdrop to compute partial derivative of J with respect to Theta
		* Implemented with a for loop `for i = 1:m`
		* Perform forward and back propagation using example (x, y) (get activations and delta terms for l=2,…,L)
	* Use gradient checking to compare partial derivative of J computed using backpropagation to the numerical estimate of the gradient of J
		* Then disable gradient checking code
	* Use gradient descent or advanced optimization method with back propagation to try to minimize J(theta) as a function of parameters Theta
		* Neural-network cost functions are non-convex, but this usually turns out not to be a problem 
		
### Developing and Improving an ML system
* Suppose you have implemented regularized linear regression to predict housing prices, however when you test your hypothesis on a new set of houses, you find that it makes unacceptably large errors in its predictions. What should you try next? -> See below section ‘Deciding what to do Next for Context’
	* Get more training examples 
	* Try a smaller set of features to prevent overfitting
	* Need additional features, since the current ones aren’t informative enough
	* Try adding polynomial features (x1^2, x1x2) 
	* Try increasing/decreasing regularization lambda
* Run a *machine learning diagnostic*, a test that you can run to gain insight on what is/isn’t working with a learning algorithm, and gain guidance as to how best to improve its performance.  Can take time to implement, but are a good use of time
* Evaluating a Hypothesis
	* A hypothesis overfits, and therefore fails to generalize new examples not in training set
	* For problems with a large of features, may be impossible to plot hypothesis 
	* Typical 70/30 split for training and test
	* Training/testing procedure for linear regression
		* Learn parameter Theta from training data (minimizing training error J)
		* Compute test error with the learned theta
		* Misclassification error (0/1 misclassification error)
			* Define the error as
				* 1 if h(x) >= 0.5, y=0, or h(x) <0.5, y=1 (incorrect)
				* 0 otherwise (correct classification)
			* test error = (1/mtest) = sum(1 to test)(error(h(x), y),  the fraction wrong
	* Once parameters were fit to some set of data (training set), the error of the parameters as measured on that data (training error J(theta)) is likely to be lower than the actual generalization error 
* Model selection
	* Of each hypothesis, run them through the model on the _test set_ and see which has the lowest Jtest.  Say we have many models with different degrees of polynomial d
		* But how well does this model generalize?
		* Suppose we come up with a chosen model: J(test(chosen)) is likely to be an optimistic estimate of the generalization error. Ie extra parameter d
		* Ie overfitting to the test set
	* Instead of splitting into just training/test, split into three:
		* Training, test, (cross) validation sets
		* Typical ratio of 60:20:20
		* Process:
			* Go through each hypothesis, train on training set, and choose the one with lowest error on val set, when that theta is finally applied to the test set, it is not biased
* Diagnosing bias vs. variance
	* On a graph of d as defined above and error 
	* Jtrain decreases, but Jcv is convex, decreasing then increasing 
		* the cross validation error will tend to *decrease* as we increase d up to a point, and then it will *increase* as d is increased, forming a convex curve.
		* That means that this is a high variance problem, where the degree of the polynomial was too large for the dataset.
	* Bias (underfit) symptoms
		* Jtrain is high
		* Jcv is also high, to around the level of Jtrain
	* Variance (overfit) symptoms
		* Jtrain is low
		* Jcv is much larger than Jtrain error
* Regularization and Bias/Variance
	* Suppose fitting a high order polynomial
		* large lambda -> underfit
		* good lambda -> just right
		* small lambda -> overfit
	* Have a range of lambdas, for example 0, 0.01, 0.02, and up in orders of 2
		* Use each lambda, and get the theta computed from each
		* Then use each theta and validate them on the validation set
		* Pick whichever has the lowest error on the validation set
	* A low lambda induces variance, while a high lambda induces bias
		* So as with d, an intermediate level of lambda is better
		
### Learning curves
* With error on y-axis and m (training set size) on x-axis, plot the mean squared error for Jtrain and Jcv
* Artificially limit to 20~40 training examples
* A symptom of high bias 
	* For example straight line fitted to concave function
	* Validation set will have very high error that only falls slightly until reaching a certain error
	* Training set will have small error that rises until reaching a certain error
	* The error that the val and train set will sometimes be similar
	* If a learning algorithm is suffering from high bias, getting more training data will not help much by itself 
* High variance
	* For example, a high polynomial function to concave data points
	* Jtrain will be low, and perhaps rise a bit
	* Jcv will be high
	* So the main identifier is that there is a large gap between Jcv and Jtrain
	* If a learning algorithm is suffering from high variance, getting more training data is likely to help.  Ie with more examples, the Jcv which is steadily falling but was truncated by the number of examples, may be improved by the number of training examples
	
### Deciding What to Do Next 
* In the menu above of how to debug a learning algorithm…
	* More training examples to fix high variance
	* Smaller set of features to fix high variance
	* Additional features to fix high bias
	* Adding polynomial features to fix high bias
	* Decreasing lambda to fix high bias
	* Increasing lambda to fix high variance
* Neural Network size
	* small network
		* fewer parameters
		* more prone to undercutting
		* computationally cheaper
	* large neural network (deep/wide or both)
		* more parameters
		* more prone to overfitting
		* computationally more expensive
		* use regularization to address overfitting, usually larger nn the better
	* Number of hidden layers/units? Usually, single hidden layer is a reasonable default
	
### ML tasks in Building a Spam Classifier
* Supervised learning
* x = features of email
* y = 1 spam or 0 not spam
* Features x: Choose 100 words indicative of spam/not spam
	* given an email, see if those words appear in the email
	* binary feature vector, of dimensions = words
* In practice instead of manually picking 100 words, pick most frequently occurring n words (10k to to 50k)
* How to spend your time to reduce error?
	* More data
	* Develop more sophisticated features based on email routing information (from email header)
	* Develop sophisticated fathers for message boy, eg are ‘discount’ and ‘discounts’ the same? ‘Deal’/‘Dealer’?
	* Detect misspellings
* It’s hard to tell in advance which of these would be the best, so best not to invest and attach too heavily to one methodology

### Error Analysis
* When starting an ML project, start with a simple algorithm that you can implement quickly. Implement it and test on cross-val data
* Plot learning curves to decide if more data, more features, etc. are likely to help
	* Very hard to what to see what is needed w/o a learning curve 
* Error analysis: Manually examine the examples (in cross val set) that algorithm made errors on. See if you spot any systematic trend in what type of examples it is making errors on.
	* For the spam classifier, manually see what emails the algorithm is misclassifying and see if there are any systematic examples that the classifier is getting wrong
* m_cv = 500 examples in cross val set
* Algorithm misclassifies 100 emails -> Manually examine the 100 errors and categorize them based on:
	* What type of email it is
	* What cues (features) you think would have helped the algorithm classify them correctly
* Perhaps after analyzing, deliberate misspellings is not as common as unusual punctuation, and pharma spam is not as common as phishing emails
* Should discount/discounts/discounted/discounted be treated as the same word? 
	* Use stemming to produce a single number that is representative of the key word
	* Be sure to double check that it does not make errors like making universe/university the same
	* Error may not be helpful for deciding if this is likely to improve performance.  The only way is to try it out.
* Do Error Analysis on the val set rather than the test set
	* Start with a quick and dirty implementation, less than 24 hours on it
	* Having a single number to measure whether the model is better or worse is very valuable as a vehicle to quickly see what is improving or worsening the model.
	
### Error Metrics for Skewed Classes
* Cancer classification with logistic regression model
	* 1% error with 99% correct diagnoses
	* But in actuality, only 0.5% of patients have cancer
		* Now the .5% error is no longer as impressive
			* An algorithm that always returns 0 would have 0.5% error
	* The number of positive examples is much much smaller than the number of negative examples.  
		* Only 0.5% of patients having cancer is an example of skewed classes, where there is a very uneven proportion of classes
	* With skewed classes, it is hard to tell with an accuracy metric if anything useful is happening. (Is 99.2 -> 99.5% accuracy meaningful?)
* For classification, it is better to use *Precision/Recall* (Confusion Matrix)
	* y = 1 in presence of rare class that we want to detect
	* True Pos / True Neg vs False Pos / False Neg
	* Precision: of all patients where predicted y=1, what fraction actually has cancer?
		* `True positives / # predicted positive`
		* = `True positives / (True pos + False pos)`
	* Recall: Of all patients that actually have cancer, what fraction did we correctly detect as having cancer?
		* `True positive / # actual positives`
		* = `True Positives / (True positives + False negatives)`
	* A model that just returns 0 will then have a low precision
		* Understand that the algorithm is actually doing well instead of ‘cheating’
	* convention is that `y=1` in presence of the rare class.  
	
### Trading off Precision and Recall
* Continuing cancer classification problem
	* Decision Boundary at 0.5
	* Suppose we want to predict y=1 (cancer) only if very confident
		* One way would be to change the decision boundary to 0.7
			* Only predict when more confident
			* End up with a classifier that has *higher precision* 
			* But *lower recall*, because y=1 because it will predict positive on a smaller number of patients
			* 0.9 would be a even higher precision classifier, but miss out on a lot of true positives
	* Avoid to avoid missing too many cases of cancer (avoid false negatives)
		* Set db to 0.3 -> More than 30% chance? Then inform
			* *Higher recall* because flagging a higher number of patients
			* But *lower precision*
	* For most classifiers there is going to be a trade off between precision and recall.  A curve between the two is plot table
* *F score*
	* How to compare precision/recall numbers?
	* The importance of one real number metric again
		* Is a Precision/Recall of 0.5/0.4 better than 0.7/0.1?
	* Average of P and R is not good, because the ‘always 0’ algorithm would have a very high recall but low precision. One with a very skewed 0.02/1.0 would have a higher average than 0.5/0.4 but be subjectively worse
	* Thus use *F Score*
		* 2 x (P x R) /  (P + R)
		* Also known as F1 score. Takes the average of the two, but gives the one with a lower score a higher weight.  

### How much Data to train on / Designing a high accuracy learning system
* Under certain conditions, getting a lot of data is good for a high performance learning algorithm
* Classify between confusable words {to, two, too}, {then, than} - 
	* /Banko and Brill, 2001/
	* I ate __ eggs <- two
	* Comparing different Algorithms: Perceptron (Logistic regression), Winnow, Memory-based, Naive Bayes
	* Conclusions
		* Most of these algorithms gave similar performance
		* From 0.1 million to a billion training examples, all algorithms monotonically increased.  Even an ‘inferior’ algorithm got better
		* This led to `It’s not who has the best algorithm that wins, it’s who has the most data`.  Which is sometimes true
* *Large data rationale*, creating the conditions for an accurate algorithm
	* Assume feature x has sufficient information to predict y accurately
	* Example: I ate __ eggs <- two, over ‘to’ and ‘too’
		* From the surrounding words, can unambiguously decide which ‘two’ to use
	* Counterexample: Predict housing price from only size (feet^2) and no other features
		* There are many other factors that would affect the price, so just using price would not produce an accurate prediction
	* *Useful test*: Given the input x, can a human expert in the domain confidently predict y?
	* Use a learning algorithm with many parameters (eg log/lin regression with many features; neural network with many hidden units)
		* ie. low bias algorithms (high complexity)
		* Jtrain(theta) will be small 
	* Use a very large training set
		* Even with a large number of parameters, will be unlikely to overfit
		* Jtrain(theta) will be close to Jtest(theta)
	* The previous two put together hopefully creates the condition for Jtest to be small.  
	* We want to not have high bias and not to have high variance
		* Bias problem is addressed by using many parameters
		* Variance is addressed by having large training set 

## Support Vector Machines	
### Optimization Objective / Support Vector Machines
* terms = theta’ * x
* What matters more than the choice of algorithm, is the data used and the application of the algorithm, i.e. choice of features, regularization parameter, etc.
* Support Vector Machines are more interpretable than NNs, and sometimes more powerful way of learning complex non-linear functions
	* Widely used within the industry and academia
* Alternative view of logistic regression
	* Given h(x) = sigmoid function and z, if y=1 we want h(x) to be close to 1, and z to be far >> 0 (i.e. when z is far above 1, then in the sigmoid function the value of g(z) will be close to the asymptote of 1
	* Conversely if y=0, want h(x) to be close to zero, and z to be <<0
* Unlike logistic regression, SVM outputs 1 or 0 directly 
	* 1 if theta’ * x >= 0, 0 otherwise
* The C parameter tells the SVM optimization how much you want to avoid misclassifying each training example.

### SVM - Large Margin Intuition
* SVMs are sometimes referred to as large margin classifiers
* Because of the shape of the SVM cost curve
	* If y =1, want z >= 1 (not just >= 0)
	* If y = 0, want z <=01 (not just <= 0)
	* Don’t want it to be barely qualify, want z to be much greater than 0.  This builds in an extra safety factor/margin into the SVM
	* SVM finds the decision boundary with the largest margin between the classes
		* This is a result of the optimization problem
* However, the large margin finding makes the classifier susceptible to outliers if C were large.
* C plays a role similar to 1/lambda.  So a large C or small lambda would result in being over influenced by outliers

### SVM Kernels
* Adapting SVMs for non-linear classifiers by using kernels
* For a non-linear decision boundary, is there a different/better choice than a higher order polynomial function?
* Using Gaussian kernels which is f = e^( - abs(x - l)**2) /(2 * sigma))
	* given x, compute new feature by proximity to landmark l , the euclidean distance
	* (x - l)**2 being the projected vector length of theta that is used by the SVM to maximize margins
	* if x were almost on top of the landmark, f will be close to 1
	* if x were far from l, then f will be close to 0
* sigma being a tuning parameter for f.  by reducing sigma, the areas in which f is close to 1 is reduced, and conversely increasing sigma means increasing the ranges for which f is close to 1
* End up with m landmarks, one for each in the training set
	* So getting the euclidean distance will tell us how close that example is from a point in the training set
* all f’s go into a feature vector, of m dimensions
* instead of using the X’s as a n+1 vector for input, instead use the m dimensional vector of f’s
* SVM Parameters
	* C = 1 / lambda
	* large C: lower bias, high variance
	* small C: higher bias, low variance
	* sigma^2
		* if sigma squared is large:  features /f/ vary more smoothly. Higher bias, lower variance. 
		* small sigma^2: features /f/ vary less smoothly, lower bias, higher variance

### Using an SVM
* Solving for theta, use SVM package (liblinear, libsvm) which are more highly optimized
* Need to specify
	* Choice of parameter C
	* Choice of kernel (similarity function)
		* No kernel (linear kernel)
			* predict y=1 if theta’ * x >= 0
			* standard linear classifier
			* large n features, m examples is small
		* Gaussian kernel
			* Choosing sigma^2 for bias/variance tradeoff
			* n is small, and/or ideally m is large
	* sometimes the SVM package may ask for your kernel similarity function
```
function f = kernel(x1, x2),
	f = exp(-((x1 - x2)**2 / (2 * sigma^2 ))
return [f1; f2; .... fm]
# note: perform feature scaling before using Gaussian kernel, otherwise the abs(x - l))^2 term will be dominated by larger values
```
* Not all similarity functions /similarity(x,l)/ make valid kernels.  Need to make sure they satisfy Mercer’s Theorem to not diverge and use all the numerical tricks
	* polynomial kernel: k(x,l) = (x’ * l)^2 , (x’ * l + 1)^3
		* (x’*l + constant) ^ degree, usually worse than Gaussian kernel and must have inputs of positive x and l
	* string kernel for strings (similarity between two strings is sine similarity), chi-square, histogram kernel, intersection kernel
* Multi-class
	* Most SVM packages have it built in
	* Otherwise use one v all method
* Logistic Regression vs SVMs
	* n = number of features
	* m = number of training examples
	* If n is large ( relative to m ) 
		* Use logistic regression, or SVM without a kernel (linear kernel)
	* If n is small (1~1000), m is intermediate (10~10000)
		* Use SVM with Gaussian kernel
	* If n is small(1~1000), m is large(m=50000+)
		* Create/add more features, then use logistic regression or SVM without a kernel (maybe outdated advice)
	* => Neural networks likely to work well for most of these, but slower to train
	* programming and problem solving with c++ dale and weems

## Unsupervised Learning
* give a data set to an algorithm and ask it to find some sort of structure in it
* one that finds discrete groupings of data would be a clustering algorithm

### K-Means Clustering
* For two clusters:
* Randomly initialize two points, ‘cluster centroids’ (because want two clusters)
* K-Means is an iterative algorithm with two steps
	* Cluster assignment
		* Go through each example, and depending on which centroids it is closer to, assign it to one of the clusters
	* Move centroid
		* Move the centroids to the average of the examples in their respective clusters
	* Repeat until reaching a convergence.  On a data set with distinct separation, even with further iterations the centroids will not move and examples will not switch their cluster group
* *Inputs*
	* K - parameter for number of clusters
	* Training set {x1, x2, …. xm}
	* x is a n-dimensional vector, by convention drop x0, x0=1
* K-means algorithm
	* Randomly initialize K cluster centroids mu1, mu1, muK into a n dimensional vector
	* Repeat
```
	for i = 1:m,
		c(i) := index (from 1 to K) of cluster centroid closest to xi
		abs( xi - muk )
	for k = 1:K,
		muk :- average of points assigned to cluster K
```

* K-means for non-separated clusters
	* Often there might be not very well separated clusters
	* K-means can still cluster out the data

### K-means Optimization Objective
* Insight into k-means, finding better clusters, and avoiding local optima
* optimization objective
	* J = (1/m) * sum(1:m) (  abs ( xi - muCi )^2 ) )  
		* muCi being the cluster centroid to which xi was assigned
		* The inner term being the abs value of the squared distance between xi and its assigned centroid
	* Also known as a “Distortion Cost Function”
	* The cluster assignment is minimizing J cost while holding the centroids fixed
	* The move centroid step chooses mu that minimizes J
	
### Random Initialization
* How to initialize K-means and avoid local optima
* Should have K < m
* Randomly pick K training examples
* Set mu1, …, mum equal to these K examples
* This means that K-means can arrive at different solutions or local optima
* -> Run K-means multiple times to ensure that the right solution is reached
* Typical to run k-means 50~1000 times
```
for i = 1 to 100
{
	Randomly initialize K-means
	Run K-means. Get c1...cm, mu1....muk
	Compute cost function (distortion)
}
```

### Choosing the Number of Clusters
* By far the most common way is to set it manually after looking at the plot or the results of an initial clustering
* Often genuinely ambiguous how many clusters you need
* ‘Elbow method’ where you choose the K-cluster number by seeing where the rate of decrease in J goes down more slowly
	* But this is not commonly used because plots where there is no obvious elbow are more common.
* Sometimes k-means is run to get clusters to use for some later/downstream purpose.  Evaluate K-means based on a metric for how well it performs for that later purpose

### Dimensionality Reduction
* Another type of unsupervised learning
	* one use is for data compression, not only to use less memory/storage space but also to speed up learning algorithms
* Collected a data set with many, many features
	* Two features are both the length of a thing, maybe not entirely linear relationship because of rounding
	* Reduce data from 2dim to 1dim, making a new feature from two highly correlated features
* 1000D -> 100D also possible
* 3D -> 2D
	* all data of x1,x2,x3 roughly lies on a plane
	* project down to a 2d
		* a 2d plane requires 2 points for each axis
		* so the representation of the points on the flat plane in a 3dim can be expressed as just a 2d graph looking down at it flat 
* In a typical setting, going from n features to k features. We expect
	* k <= n
	* k=2 or k=3 since 2d or 3d data is plot table but don’t have ways to visualize higher dimensional data. 

### Principal Component Analysis
* The most common method for dimensionality reduction
* PCA tries to find a lower dimensional surface, perhaps a line so that the error is minimized
	* This error being called the ‘projection error’, from the actual point to the surface upon which it is projected
* Standard practice to first perform mean normalization and feature scaling so that the features being reduced have a similar scale
* Problem formulation
	* Reduce from n-dimensional to k-dimensional by finding a lower dimensional surface onto which to project the data
	* Find k vectors u1,u2….uk  onto which to project the data, so as to minimize the projection error 
* PCA is not linear regression
	* linear regression attempts to minimize the mean squared error which is the straight distance from the data point to the fitted line
	* the pca error distance is projected, so the error is measured perpendicular to the line instead of straight up/down like mse
	* additionally in linreg the objective is to use x to predict y, whereas in PCA, the line is exploring the relationship between the data points
		* all features are treated equally in PCA 

### PCA Algorithm
* Training set: x1….xm
* Preprocessing 
	* feature scaling & mean normalization for all x
	* mu_j = (1/m) sum(1 to m)(x_j^i)
		* Replace each x_j^i wth x_j - mu_j
	* If different features on different scales,, scale features to have comparable range of values
		* replace x_j^i with (xji - muj) / (std dev)
* For example if transforming from 2D to 1D
	* The 2 dim vector that represents  each x point, will become a 1 dim vector that represents that point’s projected position on the PCA line, represented as `z`
	* `u` stands for the vectors that will express the dimensional surface that the points are projected to
		* a 2d -> 1d reduction will then just have a final  `u1` 2dim vector that represents the projection line
			* z will be a 1 dim vector
		* a 3d -> 2d reduction will have `u2` and `u1` to represent the 2d surface for projection.  the final z  vector will be 2dim of u1 and u2
			* z will be a 2dim vector of z1 and z2
* Compute covariance matrix
	* Sigma = (1/m) sum(1 to n) xi * xi’
		* octave: `Sigma = (1/m) * X’ * X;`
	* Sigma will be an n x n matrix
		* xi is an `n x 1` vector, and its transpose is `1 x n`  so Sigma will be nxn
* compute ‘eigenvectors’ of matrix Sigma
	* [U,S,V] = svd(Sigma); 
	* calling the octave function ‘svd’, singular value decomposition
	* same as eig() function, but more numerically stable
* U will be an n x n matrix of the u’s that we want
	* since we are reducing the data from n-dimensions to k-dimensions, take the first k columns of this matrix to get u1…uK
	* these selected columns will form the new matrix U_reduce, a `n x K` dimensional matrix 
* z = U_reduce transpose * x
	* the resultant z will be a k dimensional vector
* Summary
```
Sigma = (1/m) * X' * X;
[U, S, V] = svd(Sigma);
Ureduce = U(:, 1:k);
z= Ureduce' * X;  % keep x0, don't use x0=1
```

### Reconstruction from Compressed Representation
* Going back from the reduction to an approximation of the the original representation 
* given that `z= Ureduce' * x;`
* `x_approx = Ureduce * z`
	* x_approx will be close to whatever existed before PCA.  The approximated data points will lie along the projection surface, but that still results in a fairly good approximation of the original data. 

### Choosing the Number of Principal Components (K)
* PCA tries to minimize average squared projection error
	* which is the (1/m) * abs((x - xapprox))^2
* Total variation in the data is (1/m)(sum i to m of abs(x^i)^2)
* Typically choose k to be smallest value so that
	* (squared projection error) / (total variation) <= 0.01
		* “Retain 99% of variance”.  so the error to variation ratio was less than 1%
* Algorithm:
	* Try PCA with k=1,2,3…. 
	* Compute Reduce, z(1)…z(m),  Xapprox(1)….Xapprox(m)
	* Check if the ratio is less than or equal to 0.01
	* However this is inefficient
	* Fortunately calling [U, S, V] = svd(Sigma) gives a useful thing to use
		* S is a square zeros matrix with values in the diagonal
		* For given K, the check can be computed as 
			* `1 - (sum(i to k of Sii) / sum(i to n of Sii) <= 0.01`
			* or `(sum(i to k of Sii) / sum(i to n of Sii) >= 0.99`
		* So svd() will only need to be called once
		* Choosing a k-value, the value that the above spits out can be expressed as the percent of variance retained. 

### Advice for Applying PCA
* PCA can be used to speed up the running time of an algorithm
* Steps:
	* Given a labeled data set of (x(1), y(1))….(x(m), y(m)
	* Extract inputs, an unlabeled dataset of x(1)….x(m)
		* Transform these x’s to z’s with PCA
	* combine into the new training set of (z(1),y(1)…..(z(m), y(m))
	* then use this z in whatever h(x) as h(z) 
* Note mapping x -> z should be defined by running PCA only on the training set.
	* Only after getting the mean normalization, feature scaling, and Ureduce parameters from the training set can the mapping be applied to the training and cross val sets
* Not unusual to reduce features by 5x or 10x and still retain most of the variance
* And an algorithm using lower dimensional data will run much faster
* Choosing k=2 or 3 makes it possible to visualize high dimensional data
* One misuse of PCA - trying to prevent overfitting
	* Use z instead of x to reduce the number of features to k < n
	* Thus, fewer features, less likely to overfit
	* Better to use /regularization/
	* The reason is that PCA works without knowing ‘y’, so it might throw away some valuable information 
* PCA sometimes used where it shouldn’t be
	* Design of ML system:
		* get training set
		* /run PCA to reduce x in dimension to get z/
		* train logreg on (z, y)
		* test on test set: map x_test to z_test. Run h(z)
	* But *before* implementing PCA, first try running with the original/raw data x.  Only if that doesn’t work, then implement PCA for z

## Anomaly Detection
### Problem Motivation
* Mainly unsupervised, but there are aspects which are related to supervised learning
* For example, for an aircraft engine have two features of:  heat generated and vibration intensity.  A new engine x_test which is far away from the cluster of OK engines could be considered anomalous
	* Is /xtest/ anomalous? 
	* For this model have p(x), where p is the probability of x being OK
		* if p(xtest) < Epsilon, flag as anomalous
		* p(test) >= Epsilon should be okay
* Similarly for fraud detection
	* x(i) for features of user i’s activities
	* Model p(x) from data
	* Identify unusual users by checking which have p(x) < epsilon

### Gaussian / Normal Distribution
* NLKEZB0
* x ~ /N/(mu, sigma^2)
	* tilde means ‘distributed as’, curly N is ‘normal’
	* x distributed as normal with this mu and sigma^2 variance param
	* sigma is std dev, so sigma^2 is variance
* Probability defined as p(x; mu, sigma^2)
	* denotes that the probability of x is parameterized by mu and sigma^2
	* `p(x; mu, sigma^2) = 1/( (2*pi)^(1/2)  * sigma ) * exp( - (x - mu)^2 / 2sigma^2)`
		* just the formula for the bell shaped curve that expresses the normal distribution
* mu = 0, sigma = 1 means centered at zero, and of the regular appearance
* mu = 0, sigma = 0.5 means centered at zero, thinner and taller, since sigma is the width between the center point and the 2nd quartile(?) so it would be half as wide as sigma = 1 
* mu = (1/m) * sum(i=1 to m of x(i))
* sigma^2 = (1/m) * sum(i=1 to m of (x(i) - mu)^2)
	* These parameters are the maximum likelihood estimates of the primes of mu and sigma^2

### Anomaly Detection Algorithm
* Density estimation
	* Training set: {x(1),…x(m)}
	* Each example is x of vector dim n
		* assume x(m) ~ /N/(mu_m, sigma^2))
	* p(x) = p(x_1; mu1, sigma1^2)……p(x_n, mu_n, sigma_n^2)
		* = product(j=1 to n of p(x_j; mu_j, sigma_j^2)
* Algorithm itself
	* Choose features /x_i/ that you think might be indicative of anomalous examples
	* Fit parameters mu1,….,mu_n,   sigma1^2, ….., sigman^2
		* mu_j = (1/m) * sum(i=1 to m of x_j^i)
		* sigma_j^2 = (1/m) * sum(i=1 to m of (x_j^i - mu_j)^2)
	* Given new example x, compute p(x):
		* p(x) = (product of j=1 to n of (1/(sqrt(2pi)*sigma) * exp( - (x_j - mu_j)^2 / (2 sigma_j^2))
		* Anomaly if p(x) < epsilon
*  Set some value for epsilon
	* doing x_test for 1,2,3 and seeing where it > or  < epsilon will define a region where an example could be denied as anomalous
	
### Developing And Evaluating an Anomaly Detection System
* Once again, need a single number real-number metric to clearly evaluate and decide if it is doing better or worse
* Assume we have some labeled data, of anomalous and non-anomalous examples.
	* y=0 if normal, y=1 if anomalous
* Training set: x(1), x(2), … x(m) assume normal examples / not anomalous, although it’s alright if a few slip through
	* Also define cross-val and test set, and probably include a few examples that are known to be anomalous
* 10000 normal examples, 20 anomalous
	* Put 6000 from normal into training set
		* These examples used to generate p(x) = p(x1; mu1, sigma1^2)……
	* 2000 normal, 10 anomalous into cross val set
	* 2000 normal, 10 anomalous into test set
* Algorithm evaluation
	* Fit model p(x) on training set {x1,…,xm}
	* On a cross validation/test example x, predict
		* y = 1 if p(x) < epsilon -> anomaly
		* y = 0 if p(x) >= epsilon -> normal
	* Possible evaluation metrics:
		* not good: classification accuracy because of the skewed classes
		* true positive, false positive, false negative, true negative
		* precision / recall
		* F1-score
	* can also use cross-val set to choose parameter epsilon by choosing the value of epsilon that maximizes the F1 score

### Anomaly Detection vs Supervised Learning
* Since some of the data is labeled, why don’t we just use logistic regression or a neural network to learn the classification?
* *Comparison*
	* Anomaly detection
		* /Fraud detection, manufacturing, monitoring machines/
			* A large number of examples of positives could shift it to supervised
		* Very small number of positive examples (y=1), 0-20 is common
		* Large number of negative (y=0) examples
		* Many different ‘types’ of anomalies. Hard for any algorithm to learn from positive examples what the anomalies look like 
		* future anomalies may look nothing like any of the anomalous examples seen so far
		* So it may be more effective to just model negative examples with p(x)
		* Have such a small number of positive examples that it is not possible for a learning algorithm to learn very much from them
		* Instead take a large set of negative examples and learn p(x) from them.  Reserve the small number of positive examples for validation
	* Supervised Learning
		* /Email spam, weather prediction, cancer classification/
		* Large number of positive and negative examples
		* Enough positive examples for algorithm to get a sense of what positive examples are like, future positive examples likely to be similar to ones in training set
	* Even though are are many different types of spam, it is considered more of a supervised learning problem because there are many positive examples to learn from 

### Designing and Selecting Features for an Anomaly Detection Algorithm
* Plot the data to make sure it looks somewhat Gaussian before feeding it to the algorithm
	* It’ll actually probably still work even if the data isn’t Gaussian
	* Plot histogram with `hist()` in Octave
* If there is a long tail, maybe try transformations to even out the data
	* log transformation to a tail distribution could turn it into a gaussian distribution
	* log(x + c), play with constant `c` until it produces a suitable Gaussian distribution
	* or ` x .^ 0.01` or some other constant 
* How to come up with features for anomaly detection? 
	* Error Analysis
	* Want p(x) large for normal examples x
		* p(x) small for anomalous examples
	* Most common problem: p(x) is comparable (eg both large) for normal and anomalous examples
		* Perhaps find an anomalous example that had the high p for both, and develop a new feature from how it differs from the normal examples
* Monitoring computers in a data center
	* Choose features that might take on unusually large or small values in the event of an anomaly
	* For a data center: memory use, disk access, CPU load, network traffic
		* Perhaps consider an event where the server gets stuck in an infinite loop, where CPU load grows but network traffic remains low
		* Might be good to have a new feature of `CPU load / network traffic`
			* Or even `CPU load ^2 / network traffic`
			* These features are good to capture irregular behavior

### Multivariate Gaussian Distribution
* Can catch some anomalies that univariate didn’t.
* For features in n-dim vector, don’t model p(x1), p(x2)…. separately. Instead model p(x) in one calculation
	* Parameters: mu in n-dim vector, covariance matrix of n x n
* `p(x; mu, Sigma) = (1 / (2pi ^ n/2 * det(Sigma)^1/2) * exp((-1/2) * (x-mu)' * Sigma^-1 * (x - mu)`
	* det(Sigma) being the Octave command for finding the determinant of sigma
* Changing the covariance matrix to be smaller creates a wider shorter contour graph, larger creates a thinner and taller contour graph
* Allows the capturing of when you might think two features might be negatively or positively correlated

### Anomaly Detection with Multivariate Gaussian
* How to estimate mu and Sigma?
	* mu = (1/m) * sum( i=1 to m of x(i) )
	* Sigma = (1/m) * sum( i=1 to m of (x(i) - mu)(x(i) - mu)’ )
* 1 Fit model p(x) with the above calculated parameters
* Given a new example x, compute with above p(x) for multivariate gaussian distribution
* flag an anomaly if p(x) < epsilon
* *When to use Original vs Multivariate Gaussian*
	* Original Model, computing p(x) individually for each example
		* Manually create features to capture anomalies where x1,x2 take unusual combinations of values
			* Like creating the cpu load / memory feature
		* Computationally cheaper
		* Scales better to large m
		* Ok even if m (training set size) is small
		* More common
	* Multivariate, computing p(x) all at once
		* Automatically captures correlations between features
		* Computationally more expensive
			* Computing the big p(x) requires multiplication with the inverse of the covariance matrix.  Since it is n x n, could be very large
		* Must have m > n or else Sigma is non-invertible
			* Safe to be m > 10n
		* Able to capture unusual combinations of features
* Note: if Sigma is non-invertible, it may be that m > n was not satisfied, or there are redundant/linearly dependent features i.e. like x1 = x2, or x3 = x4+x5

### Recommender Systems
* An important application of machine learning utilized by many companies
* A system that can select by itself the best features to use

* Problem Formulation - Predicting Movie Ratings
	* User rates movies 0 to 5 stars
	* n_u = no. users
	* n_m = no.movies
	* r(i,j) = 1 if user j has rated movie i
	* y(i,j) = rating given by user j to movie i, only defined if r(i,j)=1 {0 to 5}
	* Theta(j) = parameter vector for user j
	* x(i) = feature vector for movie i, size n+1, x0 =1
	* For user j, movie i, predicted rating is `(Theta_j)’ * x(i)`
	* m(j) = # movies rated by user j
	* To learn Theta(j): 
		* min of theta for
```
 (1/(2m)) * sum(i:r(i,j)=1 of (prediction - y)^2) 
	+ lambda/(2m) * sum(sum( theta^2 ) ) 
```
* Very similar to linear regression, where we want to minimize this squared error term. 
* Can we predict, perhaps by grouping similar movies, what rating a person would give to a movie they haven’t seen?

### Content-Based Recommender Systems
* Suppose each movie has a set of features, x1 how much romance, and x2 how much action it has
	* So each movie is represented as a feature vector of interceptor x0 =1
		* [1; 0.9; 0] for example
* For each user j, learn a parameter Theta(j) which is a vector of size n+1.   Predict user j as rating movie i with `(Theta_j)’ * x(i)` stars
* Essentially applying a different copy of linear regression for each user. This user has their own theta ( a parameter vector), that we use to predict how many stars she will give the movie, given the movie features x1 and x2. Each user will have their own linear function 

### Collaborative Filtering Problem Formulation
* This algorithm employs /feature learning/, in which it can learn for itself what features to use
* Given the movie recommendation problem, but x1 and x2 are unknown.   If the movie ratings are given, and each user is able to give us their Theta, then the parameters can be inferred
	* So we have a User’s Theta and Rating, so solve for x in Theta’ * x = Rating
	* Given Thetas to learn x(i)
		* min(1/2) * sum(j:r(i,j)=1 of Theta’ *x(i) - y(i,j))^2 + lambda/2 *sum(1 to

### Collaborative Filtering Optimization Objective
* Given y,
	* The recommender system given x, allows us to estimate Theta
	* Given Theta, we can estimate X
	* This allows us to iteratively use Theta -> X -> Theta -> X…. and continuously optimize the values with each step 
* But the final form of this algorithm allows the combination of the two minimizations  into a single objective to minimize both simultaneously
	* Formula in notes and screenshots
1. Initialize x and Theta to small random values
2. Minimize J(x and Theta) using gradient descent or an advanced optimization algorithm
3. For a user with parameters Teta and a movie with (learned) features X, predict a star rating of Theta’ * X.

### Vectorization : Low Rank Matrix Factorization
* Given the grid of users, movies, and their rankings, construct a matrix Y(i,j)  where row i, col j represents a specific user’s rating for a specific movie
	* So of course each element on the matrix can be represented as (theta(j))’ * X(i)
* The simpler way would be to vectorize this 
	* Stack X into `[x(1)'; x(2)'; ... x(nm)' ]`
		* Note that they are transposed and put into their own row
	*  Theta = `[Theta(1)’;….Theta(nu)’]` (also transposed)
	* Then to get the desired final representation, multiply X by Theta transpose
* This is called low rank matrix factorization
	* Names comes from the property of being a low rank matrix in linear algebra

* An application of using the learned features from collaborative filtering is finding related movies
	* For each product i, we learn a feature vector x(i) of n dimensions
* Address the following problem: finding movies j related to movies i
	* find movies where the distance, abs(x(i) - x(j) is small
	* This distance being small means that the movies are similar, will have similar values for x1 romance, x2 action and so forth
	* Ie finding the 5 most similar movies to i, is finding the 5 j movies with the smallest abs(x(i) - x(j))

### Mean Normalization
* Normalizing each row to a mean of 0 will make the algorithm work better.
* From the i x j dimensional Y matrix, take the average of each movie (row)
	* This will create a j dim vector, one for each movie, called mu
* Then from each element, subtract its row’s average
* For user j on movie i, predict `theta(j)’ * X(i) + mu_i)`
/Suppose you have two matrices A and B where A is 5x3 and B is 3x5. Their product is C = AB, a5x5 matrix. Furthermore you have a 5x5 matrix R where every entry is 0 or 1. You want to find the sum of all elements C(i,j) for which the corresponding R(i,j) is 1, and ignore all elements C(i,j) where R(i,j)=0.  One way to do so is the following code/
```
C = A * B;
total = 0;
for i = 1:5,
	for j = 1:5,
		if (R(i,j) == 1)
			total = total + C(i,j);
		end
	end
end
```

## Large Scale Machine Learning
 Improving efficiency and architecture
* Batch GD: Use all m examples in each iteration
* Stochastic gradient descent: Use 1 example in each iteration
* Mini-batch gradient descent: Use b examples in each iteration

### Learning with Large Data Sets
* “It’s not who has the best algorithm, it’s who has the most data”
* Empirically shown that using a low-bias algorithm on a lot of data does very well
* But working with these sets is computationally expensive
	* Consider that m=100 million is common
	* Then consider that the gradient descent for linear regression requires a summation over 1 to m.  Then this summation must be computed for a single step of descent.
* Suppose a  supervised learning problem with a very large data set
	* Verify if using all of the data is likely to perform much better than just using a small subset by plotting a learning curve for a range of values of m and verify that the algorithm has high variance when m is small
		* i.e. large difference between J of cross-val and J of train
	* If the learning curves already indicate bias, then it would be unlikely to greatly improve with more training examples and sticking with the subset is alright
		* Influence towards bias by adding more features / nn nodes and layers

### Stochastic Gradient Descent
* Using SGD will allow the algorithm to scale better to large datasets
* h(x) = theta * x, Jtrain = (1/2m)*(sum from 1 to m of (h(x) - y)^2) - Linear Reg for this example
	* And this cost function is known to produce the concave bow-shaped function that has a single global optimum
* Thus if m is large, then computing the derivative of the cost function for gradient descent is difficult because it must perform the summation from 1 to m for each step .  i.e. Batch gradient descent that looks at all the examples 
* Stochastic gradient descent
	* cost(theta, (x(i), y(i))) = (1/2) (h(x) - y)^2
	* Jtrain = (1/m)(sum from i=1 to m of cost(theta, (x(i), y(i)))
	* Steps
		* 1. Randomly shuffle dataset as preprocessing
		* 2. repeat 1-10x { for i -1,…m {
						Thetaj = Thetaj - alpha(h(x) - y) * xj
						for all j
				}
* SGD is scanning through the training examples
	* Looks at the first example, takes a small gd step using just the cost of the first example
	* Then onto the second example with another small step and so on 
	* This is also a motivation for randomly shuffling the data
* Rather than waiting for the summation of all the examples, instead use a small step using one example
* SGD will general move the parameters in the direction of the global minimum, but not always.  
* SGD will not settle at the global minimum but wander around in the near vicinity, which is alright in practice
* Depending on the size of the training set, doing the SGD loop just once may not be enough, and it may require multiple passes of the m’s
	* The inner loop might only need to be done once if m is very large
* Going through all examples, SGD would have made m steps, while BGD would have only done one because of its summation 

### Mini-Batch Gradient Descent
* Sometimes can work faster than SGD
* Use b examples in each iteration, b being the mini-batch size
	* Typical choice of b=10, in range of 2-100
* For an iteration
	* Get b=10 examples, and perform a gradient descent step
	* Thetaj = Thetaj - alpha * (1/10) * (sum of k=i to i+9 of h(x(k)) - y(k)*x(k)
* Go over all examples, but in each iteration take the next b examples
* SGD VS Mini-Batch
	* Mini-batch is likely to outperform SGD only if you have a good vectorized implementation, and using a library to parallelize computations over the b examples

### Stochastic Gradient Descent Convergence
* How to ensure that SGD is running well and tune learning rate alpha?
* For Batch gradient descent, would plot optimization cost function as a function of the number of iterations
* Instead for SGD, during learning compute (theta, (x(i), y(i) before updating Theta using (x(i), y(i)) 
	* i.e. right before updating, compute how well hypothesis is doing on that training example
	* important to check before so that the training does not influence the result
* To check for SGD convergence, check every thousand iterations with plotting the costs
	* plot the average costs over the last 1000 examples processed by the algorithm, which gives a running estimate as to how well the algorithm is doing.
* Learning curve plots
	* Since SGD oscillates around the minimum, it will have small peaks and valleys
		* To smooth out the line and perhaps get a clearer visible trend, use a smaller learning rate
	* If the line is too erratic, expand the number of examples per plot point, perhaps 1000 -> 5000.  This will make the line smoother, but also delay results on how well the algorithm is doing
	* Sometimes a curve might result with severe oscillations that looks like it’s not decreasing
		* Expanding the number of examples per plot point could possibly reveal that the cost is actually decreasing 
	* If the curve is going up, then the learning rate is diverging and the learning rate should be decreased
* In most SGD implementations, the learning rate is held constant
	* If you want the SGD to actually converge on a minimum, then you can try to slowly decrease the learning rate alpha over time
	* `alpha = constant1 / (iteration # + constant 2)`
		* this makes the algorithm more finicky since these two constants have to be tuned as well

### Online Learning
* Allows the modeling of problems where there is a continuous stream of data coming in and we would like to learn from that
* Shipping service where user shops, and you offer to ship their package for some price
	* y=1 if the user decides to use your shipping service, y=0 otherwise
	* Features x capture properties of user, of origin/destination and asking price. We want to learn p(y=1 | x; Theta) to optimize price
	* Using logistic regression
* on the website, repeating forever:
	* Get (x,y) pair corresponding to user
	* Then update theta using just this (x,y):
		* Thetaj = Thetaj - alpha *( h(x) - y ) * xj
* After updating, discard that example
* Only look at one example at a time
* With a small number of users, it might be better to save away the data and train all at once
* An advantage is that this can adapt to changing user preferences
* Another problem:
	* User searches for a phone model
	* Have 100, return 10 results
	* x = features of phone, how many words in user query match name of phone, how many words in query match descriptor of phone
	* y =1 if user clicks link, y=0 otherwise
	* Learn p(y=1 | x; Theta) - predicted click-through rate
	* And therefore show user the 10 phones they are most likely to click on
	* Everytime the user searches on the site, that produces 10 x,y pairs
		* That is because for each of the results there are the features and the y of whether or not the phone was clicked on
		* For each x,y pair, run the gradient descent

### Map-Reduce and Data Parallelism
* Perhaps the problem is too large for a single computer
* Map-reduce
	* Using batch gradient descent of batch size 400
	* With 4 machines
		* Machine 1 does examples 1-100 of the batch, Machine 2 does 101 -200 and so forth. Each produces their own temporary j value to use in the final combination
		* This allows for the parallelization of the problem
	* Combine these into
		* Thetaj = Thetaj - alpha * (1/400) * ( tempj1 + tempj2 + tempj3 + tempj4)
		* This provides the same result as the gradient step formula as stated previously
	* This results in a 4x speedup minus network latencies and other slowdowns
* To use map-reduce, ask the question, /can the algorithm be expressed as a summation over the training set?/
	* It turns out many learning algorithms can be expressed as computing sums of functions over the training set
	* For example, the cost function of logistic regression and its partial derivative contain summations, and thus can be split into smaller problems for multiple machines, whose results can be sent to a central machine for combination
* Map reduce can even be applied on a single computer.
	* Since a computer can have multiple processing cores, each core can handle a  portion of the summation 
* Some linear algebra libraries automatically parallelize, given that the vectorization is good
* Applying map-reduce to train a neural network on ten machines.  In each iteration, each machine will compute forward and back propagation on 1/10 of the data to compute the derivative with respect to that 1/10 of the data 

## Photo OCR
* Description of a complex machine learning problem
* Creating a Machine Learning Pipeline
* How to allocate resources when deciding what to try next

### Problem Description and Pipeline
*  Image -> Text detection in the image -> character segmentation -> character classification
	* This is an example of a machine learning pipeline where a set of modules act on the same input to produce a desired output
	* How to break down a problem into modules?
	
### Sliding Window Classifier
* Image OCR is difficult because the boxes that must define the border of the text might have different sizes and different aspect ratios
* Scan an image by sliding/stepping over by a set amount of pixels, the box is able to centre and recognize the desired object 
	* If that box is 50x50, perhaps speed up the process by using a 100x100 box and resize it down to 50x50
* For text, y=1 if there is text within the box, and y=0 is there is not
* The segmentation step will go through the found text box, and y=1 if it finds a gap between letters, and y=0 if not
* The segmentation identification allows for each letter to be separated out individually and identify each letter

### Artificial Data Synthesis
* Since the advantages of a low bias algorithm with a large training set are known, creating artificial data can be useful to create a huge training set
* Two forms
	* Creating new data from scratch
	* Already having a small labeled training set and amplifying it into a larger one
* One way to create a letter set would be to take a large number of different fonts and put them on different backgrounds
* Introducing warping and other image distortions to amplify the data set
* Taking a clean audio clip and adding different background sounds
* Distortions introduced should be representative of the type of noise/distortions in the test set. 
	* It usually does not help to add purely random/meaningless noise to the data
* If you were to make one copy of each example, so that there were now two duplicates of each example to double the training set, that would end up with the same parameters Theta
* Getting more data
1. Make sure you have a low bias classifier before expending the effort (plot learning curves).  Keep increasing the number of features/number of hidden units in neural network until you have a low bias classifier
2. Consider how much work would it take to get 10x data as we currently have.  It’s commonly not that hard, so that will be very useful 
	* Consider artificial data synthesis
	* Collect/label yourself
	* Crowdsource it

### Ceiling Analysis
* Which part of the pipeline to work on next?
* Which module of the pipeline should you spend the most time trying to improve?
* Important to have a single metric for the overall system, such as accuracy for the text OCR problem
* Give the first step labeled data (i.e. give it the next module perfectly labeled/identified data), and see the new performance
	* Then continue on with the following steps 
* This allows seeing which modules give the best benefit or problem areas
