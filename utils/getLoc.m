function [centroidLoc] = getLoc(bbox)
% Copyright 2015-2016 The MathWorks, Inc.

centroidLoc(:,1) = bbox(:,1) + (bbox(:,3)/2);
centroidLoc(:,2) = bbox(:,2) + (bbox(:,4)/2);

end