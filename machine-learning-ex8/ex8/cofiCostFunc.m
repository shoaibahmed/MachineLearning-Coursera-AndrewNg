function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% Movie i
% for i = 1 : size(R, 1)
%     % User j
%     for j = 1 : size(R, 2)
%         % If rating is present for this movie i, user j pair
%         if R(i, j)
%             pred = Theta(j, :) * X(i, :)';
%             diff = pred - Y(i, j);
%             J = J + (diff .^ 2);
%             
%             % Gradients
%             X_grad(i, :) = X_grad(i, :) + diff * Theta(j, :);
%             Theta_grad(j, :) = Theta_grad(j, :) + diff * X(i, :);
%         end
%     end
% end

% Vectorized implementation
pred = X * Theta';
diff = pred - Y;
diff = diff .* R; % Count for only elements which have ratings avaiable
J = J + sum(sum(diff .^ 2));

% Gradients
X_grad = diff * Theta;
Theta_grad = diff' * X;

J = 0.5 * J;

% Add regularization
J = J + (lambda / 2) * sum(sum(Theta .^ 2)) + (lambda / 2) * sum(sum(X .^ 2));

X_grad = X_grad + (lambda * X);
Theta_grad = Theta_grad + (lambda * Theta);

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
