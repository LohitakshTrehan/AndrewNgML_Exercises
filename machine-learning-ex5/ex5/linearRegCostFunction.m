function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h_val = X*theta;
J=(1/(2*m))*sum((h_val-y).^2);
new_theta=theta;
new_theta(1)=0;
reg=(lambda/(2*m))*sum(new_theta.^2);
J = J+reg;
grad=(1/m)*((X')*(h_val-y));
reg_grad=new_theta*lambda/m;
grad = grad + reg_grad;










% =========================================================================

grad = grad(:);

end
