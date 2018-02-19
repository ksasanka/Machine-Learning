function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
thetareg=theta;
thetareg(1,1)=0;

htheta=sigmoid(theta'*X');

J1=y'*(log(htheta)');
J2=(ones(size(y'))-y')*(log(ones(size(htheta))-htheta)');
    J=-1*(J1+J2)/m + (lambda/(2*m))*(thetareg'*thetareg);
grad=1/m*(htheta*X-y'*X)+(lambda/m)*thetareg';
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
