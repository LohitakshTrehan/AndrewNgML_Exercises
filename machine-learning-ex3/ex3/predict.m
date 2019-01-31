function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
    X = [ones(size(X, 1), 1) X];
    ans1 = X* (Theta1');
    ans1 = sigmoid(ans1);
    %ans1 is m X 25 matrices of a's
    ans1 = [ones(size(ans1, 1), 1) ans1];
    ans2 = ans1*(Theta2');
    ans2 = sigmoid(ans2);
    [temp result]  = max(ans2');
    p = result';
% =========================================================================


end
