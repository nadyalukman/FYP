%% Referenced codes:

% Motion-Based Multiple Object Tracking
% Copyright 2014 The MathWorks, Inc.

% Cuixing: Yolov3 Yolov4 MATLAB
% https://github.com/cuixing158/yolov3-yolov4-matlab

%% YOLOv3 Kalman + SORT tracker with Camera Input

% The association of detections to the same object is based solely on
% motion. The motion of each track is estimated by a Kalman filter. The
% filter is used to predict the track's location in each frame, and
% determine the likelihood of each detection being assigned to each 
% track.
%
% Track maintenance becomes an important aspect of this example. In any
% given frame, some detections may be assigned to tracks, while other
% detections and tracks may remain unassigned. The assigned tracks are
% updated using the corresponding detections. The unassigned tracks are 
% marked invisible. An unassigned detection begins a new track. 
%
% Each track keeps count of the number of consecutive frames, where it
% remained unassigned. If the count exceeds a specified threshold, the
% example assumes that the object left the field of view and it deletes 
% the track.  
%
% For more information please see
% <docid:vision_ug#buq9qny-1 Multiple Object Tracking>.
%
% This example is a function with the main body at the top and helper
% routines in the form of nested functions.

function YOLOv3_KalmanSORT_Cam()
        
        addpath('CustomLayers/','utils/');

        cfg_file = 'cfg/yolov3.cfg';
        weight_file = 'weights/yolov3.weights';
        throushold = 0.5;
        NMS = 0.8;

        %% import all classes
        fid = fopen('coco.names','r');
        names = textscan(fid, '%s', 'Delimiter',{'   '});
        fclose(fid);
        classesNames = categorical(names{1});
        % Create System objects used for reading video, detecting moving objects,
        % and displaying the results.
        obj = setupSystemObjects();
        
        tracks = initializeTracks(); % Create an empty array of tracks.
        
        nextId = 1; % ID of the next track
        
        frame = snapshot(obj.reader);
        step(obj.videoPlayer, frame);
        % Detect moving objects, and track them across video frames.
        %while hasFrame(obj.reader)
        while obj.videoPlayer.isOpen()
            frame = snapshot(obj.reader);
            width = size(frame,2)/416;
            height = size(frame,1)/416;
            frame1 = imadjustn(imresize(im2single(frame),[416 416]));
            [centroids, bboxes, labels] = detectObjects(frame1);
            predictNewLocationsOfTracks();
            [assignments, unassignedTracks, unassignedDetections] = ...
                detectionToTrackAssignment();
            
            updateAssignedTracks();
            updateUnassignedTracks();
            deleteLostTracks();
            createNewTracks();
            
            displayTrackingResults();
end


%% Create System Objects
% Create System objects used for reading the video frames, detecting
% foreground objects, and displaying results.

    function obj = setupSystemObjects()
        % Initialize Video I/O
        % Create objects for reading a video from a file, drawing the tracked
        % objects in each frame, and playing the video.
        
        % Create a video reader.
        obj.reader = webcam('Logitech');
        
        obj.videoPlayer = vision.VideoPlayer;

    end

%% Initialize Tracks
% The |initializeTracks| function creates an array of tracks, where each
% track is a structure representing a moving object in the video. The
% purpose of the structure is to maintain the state of a tracked object.
% The state consists of information used for detection to track assignment,
% track termination, and display. 
%
% The structure contains the following fields:
%
% * |id| :                  the integer ID of the track
% * |bbox| :                the current bounding box of the object; used
%                           for display
% * |kalmanFilter| :        a Kalman filter object used for motion-based
%                           tracking
% * |age| :                 the number of frames since the track was first
%                           detected
% * |totalVisibleCount| :   the total number of frames in which the track
%                           was detected (visible)
% * |consecutiveInvisibleCount| : the number of consecutive frames for 
%                                  which the track was not detected (invisible).
%
% Noisy detections tend to result in short-lived tracks. For this reason,
% the example only displays an object after it was tracked for some number
% of frames. This happens when |totalVisibleCount| exceeds a specified 
% threshold.    
%
% When no detections are associated with a track for several consecutive
% frames, the example assumes that the object has left the field of view 
% and deletes the track. This happens when |consecutiveInvisibleCount|
% exceeds a specified threshold. A track may also get deleted as noise if 
% it was tracked for a short time, and marked invisible for most of the 
% frames.        

    function tracks = initializeTracks()
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
    end

%% Detect Objects

    function [centroids, bboxes, labels] = detectObjects(frame)
        
        outFeatures = yolov3v4Predict(cfg_file,weight_file,frame,width,height);
        
        scores = outFeatures(:,5);
        outFeatures = outFeatures(scores>throushold,:);
    
        allBBoxes = outFeatures(:,1:4);
        allScores = outFeatures(:,5);
        [maxScores,indxs] = max(outFeatures(:,6:end),[],2);
        allScores = allScores.*maxScores;
        allLabels = classesNames(indxs);
        
        [bboxes,labels] = selectStrongestBboxMulticlass2(allBBoxes,allScores,allLabels,...
            'RatioType','Min','OverlapThreshold',NMS);
        
        for i = 1:length(labels) 
            if labels(i)~='person'
                bboxes(i,:)=zeros(1,4);
                labels(i)= '0';
                i=i+1;
            end
        end
        bboxes(all(bboxes==0,2),:)=[];
        labels = labels(labels~='0');
        centroids = getLoc(bboxes);
    end

%% Predict New Locations of Existing Tracks
% Use the Kalman filter to predict the centroid of each track in the
% current frame, and update its bounding box accordingly.

    function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;
            % Predict the current location of the track.
            predictedCentroid = predict(tracks(i).kalmanFilter);
                        
            % Shift the bounding box so that its center is at 
            % the predicted location.
            predictedCentroid = int32(predictedCentroid)-int32(bbox(:,3:4)/2);
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
    end

%% Assign Detections to Tracks
% Assigning object detections in the current frame to existing tracks is
% done by minimizing cost. The cost is defined as the negative
% log-likelihood of a detection corresponding to a track.  
%
% The algorithm involves two steps: 
%
% Step 1: Compute the cost of assigning every detection to each track using
% the |distance| method of the |vision.KalmanFilter| System object(TM). The 
% cost takes into account the Euclidean distance between the predicted
% centroid of the track and the centroid of the detection. It also includes
% the confidence of the prediction, which is maintained by the Kalman
% filter. The results are stored in an MxN matrix, where M is the number of
% tracks, and N is the number of detections.   
%
% Step 2: Solve the assignment problem represented by the cost matrix using
% the |assignDetectionsToTracks| function. The function takes the cost 
% matrix and the cost of not assigning any detections to a track.  
%
% The value for the cost of not assigning a detection to a track depends on
% the range of values returned by the |distance| method of the 
% |vision.KalmanFilter|. This value must be tuned experimentally. Setting 
% it too low increases the likelihood of creating a new track, and may
% result in track fragmentation. Setting it too high may result in a single 
% track corresponding to a series of separate moving objects.   
%
% The |assignDetectionsToTracks| function uses the Munkres' version of the
% Hungarian algorithm to compute an assignment which minimizes the total
% cost. It returns an M x 2 matrix containing the corresponding indices of
% assigned tracks and detections in its two columns. It also returns the
% indices of tracks and detections that remained unassigned. 

    function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()
        
        nTracks = length(tracks);
        nDetections = size(centroids, 1);
        
        % Compute the cost of assigning each detection to each track.
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end
        
        % Solve the assignment problem.
        costOfNonAssignment = 100;
        [assignments, unassignedTracks, unassignedDetections] = ...
            matchpairs(cost, costOfNonAssignment);
    end

%% Update Assigned Tracks
% The |updateAssignedTracks| function updates each assigned track with the
% corresponding detection. It calls the |correct| method of
% |vision.KalmanFilter| to correct the location estimate. Next, it stores
% the new bounding box, and increases the age of the track and the total
% visible count by 1. Finally, the function sets the invisible count to 0. 

    function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);
            
            % Correct the estimate of the object's location
            % using the new detection.
            correct(tracks(trackIdx).kalmanFilter, centroid);
            
            % Replace predicted bounding box with detected
            % bounding box.
            tracks(trackIdx).bbox = bbox;
            
            % Update track's age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;
            
            % Update visibility.
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
    end

%% Update Unassigned Tracks
% Mark each unassigned track as invisible, and increase its age by 1.

    function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
    end

%% Delete Lost Tracks
% The |deleteLostTracks| function deletes tracks that have been invisible
% for too many consecutive frames. It also deletes recently created tracks
% that have been invisible for too many frames overall. 

    function deleteLostTracks()
        if isempty(tracks)
            return;
        end
        
        invisibleForTooLong = 30;
        ageThreshold = 10;
        
        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        
        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
        
        % Delete lost tracks.
        tracks = tracks(~lostInds);
    end

%% Create New Tracks
% Create new tracks from unassigned detections. Assume that any unassigned
% detection is a start of a new track. In practice, you can use other cues
% to eliminate noisy detections, such as size, location, or appearance.

    function createNewTracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);
        
        for i = 1:size(centroids, 1)
            
            centroid = centroids(i,:);
            bbox = bboxes(i, :);
            
            % Create a Kalman filter object.
            kalmanFilter = configureKalmanFilter('ConstantAcceleration', ...
                centroid, 100*ones(1,3), [25, 10, 1], 100);
                %                 centroid, [200, 50], [100, 25], 100);
            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);
            
            % Add it to the array of tracks.
            tracks(end + 1) = newTrack;
            
            % Increment the next id.
            nextId = nextId + 1;
        end
    end

%% Display Tracking Results
% The |displayTrackingResults| function draws a bounding box and label ID 
% for each track on the video frame 

    function displayTrackingResults()
%         % Convert the frame  to uint8 RGB.
        frame = im2uint8(frame);
%         
       minVisibleCount = 8;
       if ~isempty(tracks)
% %               
% %             % Noisy detections tend to result in short-lived tracks.
% %             % Only display tracks that have been visible for more than 
% %             % a minimum number of frames.
           reliableTrackInds = ...
               [tracks(:).totalVisibleCount] > minVisibleCount;
           reliableTracks = tracks(reliableTrackInds);
% %             
% %             % Display the objects. If an object has not been detected
% %             % in this frame, display its predicted bounding box.
           if ~isempty(reliableTracks)
% % %                 Get bounding boxes.
               bboxes = cat(1, reliableTracks.bbox);
% %                 
% %                 % Get ids.
               ids = int32([reliableTracks(:).id]);
%                 
%                 % Create labels for objects indicating the ones for 
%                 % which we display the predicted rather than the actual 
%                 % location.
               labels = cellstr(int2str(ids'));
               predictedTrackInds = ...
                   [reliableTracks(:).consecutiveInvisibleCount] > 0;
               isPredicted = cell(size(labels));
               isPredicted(predictedTrackInds) = {' predicted'};
               labels = strcat(labels, isPredicted);
                
                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);
           end
        end
        
        obj.videoPlayer.step(frame);        
    end

%% Summary
% This example created a motion-based system for detecting and
% tracking multiple moving objects. Try using a different video to see if
% you are able to detect and track objects. Try modifying the parameters
% for the detection, assignment, and deletion steps.  
%
% The tracking in this example was solely based on motion with the
% assumption that all objects move in a straight line with constant speed.
% When the motion of an object significantly deviates from this model, the
% example may produce tracking errors. Notice the mistake in tracking the
% person labeled #12, when he is occluded by the tree. 
%
% The likelihood of tracking errors can be reduced by using a more complex
% motion model, such as constant acceleration, or by using multiple Kalman
% filters for every object. Also, you can incorporate other cues for
% associating detections over time, such as size, shape, and color. 


end
