function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    temp = X * theta;
    error = temp - y;
    newX = error' * X;
    theta = theta - ((alpha/m) * newX');
    
    %theta_z = theta(1,1) - (alpha *(1/m)*  (((X(:,1))')*((X*theta)-y))  );
    %theta_one = theta(2,1) - (alpha *(1/m)*  (((X(:,2))')*((X*theta)-y))  );
    %theta_two = theta(3,1) - (alpha *(1/m)*  (((X(:,3))')*((X*theta)-y))  );
    %theta(1,1) = theta_z;
    %theta(2,1) = theta_one;
    %theta(3,1) = theta_two;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
