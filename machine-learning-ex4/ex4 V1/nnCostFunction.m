function [J grad] = nnCostFunction(nn_params,
                                   input_layer_size,
                                   hidden_layer_size,
                                   num_labels,
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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),
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


X = [ones(m, 1) X];

hid_layer=sigmoid((Theta1*X')');
hid_layer = [ones(m, 1) hid_layer];

out_layer=(Theta2*hid_layer')';
A=sigmoid(out_layer);



for k=1:num_labels
  for c=1:m
  if y(c,1)==k
  t(c,1)=1;
  else t(c,1)=0;
  endif
  end
 htheta=A(:,k);
 J1=t'*log(htheta);
 J2=(ones(size(t'))-t')*(log(ones(size(htheta))-htheta));
 J=J-1*(J1+J2)/m ;
  
end


thetareg1=Theta1;
thetareg1(:,1)=0;

thetareg2=Theta2;
thetareg2(:,1)=0;
theta_reg=[thetareg1(:) ; thetareg2(:)];



J=J+lambda/(2*m)*theta_reg'*theta_reg;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = zeros(size([Theta1(:) ; Theta2(:)]));
Y=zeros(m,num_labels);

Delta2=zeros(size(Theta2));
Delta1=zeros(size(Theta1));

t=1;
for t=1:num_labels
y1=(y==t);
Y(:,t)=y1;
end

i=1;
for i=1:m
X1=X(i,:);
Y1=Y(i,:);
z2=(Theta1*X1')';
a2=sigmoid(z2);
a2 = [ones(1, 1) a2];
z3=a2*Theta2';
a3=sigmoid(z3);
d3=a3-Y(i,:);
d2=(Theta2'*d3').*(a2.*(1-a2))';
Delta2 = Delta2 + d3'*a2; 
Delta1 = Delta1 + d2(2:end) * X1; 

end
Delta1=Delta1/m+(lambda*thetareg1)/m;
Delta2=Delta2/m+(lambda*thetareg2)/m;

grad=[Delta1(:) ; Delta2(:)];
    
end


