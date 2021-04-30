%selectStrongestBboxMulticlass Select strongest multiclass bounding boxes from overlapping clusters.
%   selectedBboxes = selectStrongestBboxMulticlass(bboxes,scores,labels)
%   returns selected bounding boxes that have a high confidence score. The
%   function uses greedy non-maximal suppression (NMS) to eliminate
%   overlapping bounding boxes only if they have the same class label. The
%   selected boxes are returned in selectedBboxes.
%
%   Inputs
%   ------
%   bboxes - an M-by-4 or M-by-5 matrix representing axis-aligned or
%            rotated rectangles, respectively. Each row in bboxes defines
%            one bounding box. See <a href="matlab:helpview(fullfile(docroot,'toolbox','vision','vision.map'),'selectStrongestBboxMulticlassBoxFormats')">supported bounding box formats</a> for more
%            information on the format used to define a bounding box.
%
%   scores - an M-by-1 vector of scores corresponding to the input bounding
%            boxes.
%
%   labels - is an M-by-1 vector of categorical or numeric labels
%            corresponding to the input bounding boxes.
%
%   [..., selectedScores, selectedLabels, index] = selectStrongestBboxMulticlass(...)
%   additionally returns the scores, labels, and index associated with the
%   selected bounding boxes.
%
%   [...] = selectStrongestBboxMulticlass(..., Name, Value) specifies
%   additional name-value pairs described below:
%
%   'RatioType'         A string, 'Union' or 'Min', specifying the
%                       denominator of bounding box overlap ratio.
%                       See bboxOverlapRatio for detailed explanation of
%                       the ratio definition.
%
%                       Default: 'Union'
%
%   'OverlapThreshold'  A scalar from 0 to 1. All bounding boxes around a
%                       reference box are removed if their overlap ratio
%                       is above this threshold.
%
%                       Default: 0.5
%
%   'NumStrongest'      Specify the maximum number of strongest boxes to
%                       select as a positive scalar or inf. Lower this
%                       value to reduce processing time when you have a
%                       priori knowledge about the maximum number of boxes.
%                       Set the value to Inf to select all the strongest,
%                       non-overlapping, bounding boxes.
%
%                       When the labels input is a categorical, you can
%                       also specify a vector that contains the maximum
%                       number of strongest boxes for each category in the
%                       labels input. The length of the specified vector
%                       must equal the number of categories in labels.
%
%                       Default: Inf (select all the strongest boxes) 
%
%  Class Support
%  -------------
%  bboxes, scores, and labels must be real, finite, and nonsparse. They can
%  be uint8, int8, uint16, int16, uint32, int32, single or double. In
%  addition, labels can be categorical. OverlapThreshold can be single or
%  double. Class of index output is double. Class of selectedBboxes is the
%  same as that of bbox input. Class of selectedScores is the same as that
%  of score input. Class of selectedLabels is the same as that of labels
%  input.
%
%  Example
%  -------
%  % Create detectors using two different models. These will be used to
%  % generate multiclass detection results.
%  detectorInria = peopleDetectorACF('inria-100x41' );
%  detectorCaltech = peopleDetectorACF('caltech-50x21');
%
%  % Apply the detectors.
%  I = imread('visionteam1.jpg');
%  [bboxesInria, scoresInria] = detect(detectorInria,I,'SelectStrongest',false);
%  [bboxesCaltech, scoresCaltech] = detect(detectorCaltech,I,'SelectStrongest',false);
%
%  % Create categorical labels for each the result of each detector.
%  labelsInria = repelem("inria",numel(scoresInria),1);
%  labelsInria = categorical(labelsInria,{'inria','caltech'});
%  labelsCaltech = repelem("caltech",numel(scoresCaltech),1);
%  labelsCaltech = categorical(labelsCaltech,{'inria','caltech'});
%
%  % Combine results from all detectors to for multiclass detection results.
%  allBBoxes = [bboxesInria;bboxesCaltech];
%  allScores = [scoresInria;scoresCaltech];
%  allLabels = [labelsInria;labelsCaltech];
%
%  % Run multiclass non-maximal suppression
%  [bboxes, scores, labels] = selectStrongestBboxMulticlass(...
%       allBBoxes,allScores,allLabels,...
%       'RatioType','Min','OverlapThreshold',0.65);
%
%  % Annotate detected people
%  annotations = string(labels) + ": " + string(scores);
%  I = insertObjectAnnotation(I, 'rectangle', bboxes, annotations);
%  figure
%  imshow(I)
%  title('Detected people, scores, and labels')
%
%  See also bboxOverlapRatio, selectStrongestBbox.

% Copyright 2017-2020 The MathWorks, Inc.

function [selectedBbox,selectedLabel] = ...
    selectStrongestBboxMulticlass2(bbox, score, label, varargin)


%#codegen
%#ok<*EMCLS>
%#ok<*EMCA>

nargoutchk(0,4)

[ratioType,overlapThreshold, numStrongest] = iParseInputs(bbox,score,label,varargin{:});

if isempty(bbox)
    selectedBbox = bbox;
    selectedScore = score;
    selectedLabel = label;
    index = [];
    return;
end

if strncmpi(ratioType, 'Union', 1)
    isDivByUnion = true;
else
    isDivByUnion = false;
end

if ~isfloat(bbox)
    inputBbox = single(bbox);
else
    inputBbox = bbox;
end

% Convert labels to numeric values.
inputLabel = iCategoricalLabelsToNumeric(label, class(inputBbox));

% Sort the bbox according to the score.
if coder.gpu.internal.isGpuEnabled
    [~, ind] = gpucoder.sort(score, 'descend');
else
    [~, ind] = sort(score, 'descend');
end

% Reorder boxes and labels based on scores.
inputBbox  = inputBbox(ind, :);
inputLabel = inputLabel(ind,:);

isCodegen = ~isempty(coder.target);
switch size(inputBbox,2)
    case 4
        % axis-aligned
        selectedIndex = iOverlapSuppressionAxisAligned(...
            inputBbox, inputLabel, overlapThreshold, isDivByUnion, numStrongest, isCodegen);
    case 5
        % rotated rectangle
        selectedIndex = iOverlapSuppressionRotatedRect(...
            inputBbox, inputLabel, overlapThreshold, isDivByUnion, numStrongest, isCodegen);
    otherwise
        % This code path is required for codegen support when all
        % dimensions of bbox are variable size. Runtime checks exist to
        % ensure the size of bbox is 4 or 5.
        selectedIndex = [];
        
end

% Reorder the indices back to the pre-sorted order.
index = coder.nullcopy(selectedIndex);
index(ind) = selectedIndex;

selectedBbox = bbox(index, :);
selectedLabel = label(index);
   
% Return an index list instead of logical vector.
index = find(index);

end

%--------------------------------------------------------------------------
function selectedIndex = iOverlapSuppressionAxisAligned(...
    bbox, label, threshold, isDivByUnion, numStrongest, isCodegen)

if isCodegen
    selectedIndex = vision.internal.detector.selectStrongestBboxCodegen(...
        bbox, threshold, isDivByUnion, numStrongest, label);
else
    selectedIndex = visionBboxOverlapSuppression(bbox, ...
        threshold, isDivByUnion, numStrongest, label);
end
end

%--------------------------------------------------------------------------
function selectedIndex = iOverlapSuppressionRotatedRect(...
    bbox, label, threshold, isDivByUnion, numStrongest, isCodegen)

if ~isCodegen
    [x,y] = vision.internal.bbox.bbox2poly(bbox);
    
    xlim = [min(x,[],'all') max(x,[],'all')];
    ylim = [min(y,[],'all') max(y,[],'all')];
    
    selectedIndex = visionRotatedBBoxOverlapSuppression(...
        x, y, label, threshold, isDivByUnion, numStrongest, xlim, ylim);
else
    selectedIndex = [];
end

end

%--------------------------------------------------------------------------
function iCheckInputBboxScoreAndLabel(bbox,score,label)
vision.internal.detector.selectStrongestValidation.checkInputBboxAndScore(bbox, score, mfilename);
iCheckLabel(label)

coder.internal.errorIf(~isempty(label) && (numel(label) ~= size(bbox,1)) ,...
    'vision:visionlib:unmatchedBboxAndLabel');
end

%--------------------------------------------------------------------------
function iCheckLabel(value)

if isa(value,'categorical')
    validateattributes(value,{'categorical'}, {'size',[NaN, 1]}, ...
        mfilename, 'label', 3);
else
    validateattributes(value,{'uint8', 'int8', 'uint16', 'int16', 'uint32', ...
        'int32', 'double', 'single'}, {'real','nonsparse','finite','integer','size',[NaN, 1]}, ...
        mfilename, 'label', 3);
end
end

%--------------------------------------------------------------------------
function labels = iCategoricalLabelsToNumeric(labels,class)
labels = cast(labels,class);
end

%--------------------------------------------------------------------------
function [ratioType, overlapThreshold, numStrongest] = ...
    iParseInputs(bbox,score,label,varargin)
% Parse and check inputs
iCheckInputBboxScoreAndLabel(bbox, score, label);
if isempty(coder.target)
    [ratioType, overlapThreshold, numStrongest] = ...
        vision.internal.detector.selectStrongestValidation.parseOptInputs(varargin{:});
else
    [ratioType, overlapThreshold, numStrongest] = ...
        vision.internal.detector.selectStrongestValidation.parseOptInputsCodegen(varargin{:});
end

if ~isempty(varargin)
    iValidateParams(ratioType, overlapThreshold, numStrongest);
end
              
isVector = numel(numStrongest) > 1;
isCategoricalLabels = isa(label,'categorical');

% When numStrongest is a vector, label must be a categorical.
coder.internal.errorIf(isVector && ~isCategoricalLabels,...
    'vision:selectStrongestBbox:mustBeCategoricalForNumStrongestVector');

% Issue a compile time error if labels do not have constant categories.
% This happens if labels are created without providing a value set when
% creating the categorical values: categorical(data). 
if isCategoricalLabels && ~isempty(coder.target) 
    cats = categories(label);
    eml_invariant(eml_is_const(numel(cats)), ...
                  eml_message('vision:selectStrongestBbox:categoriesMustBeConst'));
end

% When numStrongest is a vector, the length of the vector must equal the
% number of categories in label.
coder.internal.errorIf(isVector && numel(categories(label)) ~= numel(numStrongest),...
    'vision:selectStrongestBbox:numelNumStrongestMustEqualNumCategories');

numStrongest = double(numStrongest);
end

%--------------------------------------------------------------------------
function iValidateParams(ratioType, overlapThreshold, numStrongest)
vision.internal.detector.selectStrongestValidation.checkOverlapThreshold(overlapThreshold,mfilename)
vision.internal.detector.selectStrongestValidation.checkRatioType(ratioType);
vision.internal.detector.selectStrongestValidation.checkNumStrongestScalarOrVector(numStrongest);
end