% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (c) 2018 Pascal Monasse <pascal.monasse@enpc.fr>

function y = my_randsample(n,k)
%MY_RANDSAMPLE A replacement for randsample from the statistics
%package. The difference is that the returned vector is always ordered.
%
% Returns a k-by-1 vector of values sampled uniformly at random,
% without replacement, from the integers 1 to n.

assert(k<=n, 'Error in my_randsample: k must not exceed n')

y = [];
for i=0:k-1
    assert(size(y,2)==i)
    j = randi(n-i); % only n-i integers not picked yet
    % find j-th element in 1:n \setminus y
    l = 1;
    while l<=i && y(l)<=j
        l=l+1; j=j+1;
    end
    % insert j in vector y
    y = [y(1:l-1) j y(l:end)];
end

y=y';

end
