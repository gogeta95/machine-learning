function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

temp = zeros(size(y,1),size(num_labels,1));
for i=1:size(y,1)
  temp(i,y(i)) =1;
endfor

y=temp';

a_one =[ones(m,1) X];

z2= a_one*Theta1';
a_two = sigmoid(z2);
a_two =[ones(size(a_two,1),1) a_two];

h = sigmoid(a_two*Theta2');



h= h';


  J = sum(y.*log(h)) + sum((1-y).*log(1-h));  

  J=-J;
  J= sum(J)/m;
  

theta1 = Theta1(:,2:end);
theta2 = Theta2(:,2:end);

theta1=theta1.*theta1;
theta2=theta2.*theta2;


J = J + (lambda/(2*m)) * (sum(sum(theta1))+ sum(sum(theta2)));

D1 =0;
D2=0;
for t=1:m
a1 = [1 X(t,:)];
z2= a1*Theta1';
a2 = [1 sigmoid(z2)];


z3= a2*Theta2';
a3 = sigmoid(z3)';


d3 = (a3-y(:,t));
d2= Theta2'*d3;
d2= d2(2:end,:);
d2= d2.*sigmoidGradient(z2');

D2 = D2 + d3*a2;
D1= D1 + d2*a1;
endfor

Theta1_grad = D1/m;
Theta2_grad = D2/m;
% -------------------------------------------------------------
Theta1_grad = Theta1_grad + (lambda/m)*[zeros(size(Theta1(:,2:end),1),1) Theta1(:,2:end)];

Theta2_grad = Theta2_grad + (lambda/m)*[zeros(size(Theta2(:,2:end),1),1) Theta2(:,2:end)];
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
