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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


temp = (lambda/2).*(sum(theta.^2)-(theta(1).^2));
z = X*theta;
htheta = sigmoid(z);
func = -y.*log(htheta) -(1 - y).*log(1 - htheta);
J = (func'*ones(m,1))+temp;
J = (1/m).*J;
z = sigmoid(X*theta)-y;
z = z'*X;
z = (1/m).*(z'+lambda.*theta);
z(1) = z(1) - (lambda/m).*theta(1);
grad = z;



% =============================================================

end
