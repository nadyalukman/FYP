%% Referenced codes:

% Motion-Based Multiple Object Tracking
% Copyright 2016 The MathWorks, Inc.

% Cuixing: Yolov3 Yolov4 MATLAB
% https://github.com/cuixing158/yolov3-yolov4-matlab

%% YOLOv3 Camera Input with Kalman Filter Tracker
% Detection of moving objects and motion-based tracking are important 
% components of many computer vision applications, including activity
% recognition, traffic monitoring, and automotive safety.  The problem 
% of motion-based object tracking can be divided into two parts:
%
% # Detecting moving objects in each frame 
% # Associating the detections corresponding to the same object over time 
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

function YOLOv3_KalmanTracker_Cam()
        
        % add libraries
        addpath('CustomLayers/','utils/');
        
        % import pretrained weight files
        cfg_file = 'cfg/yolov3.cfg';
        weight_file = 'weights/yolov3.weights';
        throushold = 0.5; % set IOU threshold value
        NMS = 0.5; % set non-maximal suppression value

        % import all classes
        fid = fopen('coco.names','r');
        names = textscan(fid, '%s', 'Delimiter',{'   '});
        fclose(fid);
        classesNames = categorical(names{1});
        
        % Create System objects used for reading video, 
        % detecting moving objects, and displaying the results.
        obj = setupSystemObjects();
        
        tracker = multiObjectTracker(...
            'FilterInitializationFcn', @initDemoFilter, ...
            'AssignmentThreshold', 30, ...
            'DeletionThreshold', 22, ...
            'ConfirmationThreshold', [6 10] ...
            );
        
    function filter = initDemoFilter(detection)
        % Initialize a Kalman filter for this example.
        
        % Define the initial state.
        state = [detection.Measurement(1); 0; detection.Measurement(2); 0];
        
        % Define the initial state covariance.
        stateCov = diag([50, 50, 50, 50]);
        
        % Create the tracking filter.
        filter = trackingKF('MotionModel', '2D Constant Velocity', ...
            'State', state, ...
            'StateCovariance', stateCov, ...
            'MeasurementNoise', detection.MeasurementNoise(1:2,1:2) ...
            );
    end

    frameCount = 0;
        % Detect moving objects, and track them across video frames.
        while obj.videoPlayer.isOpen()
            frameCount = frameCount + 1;
            frame = snapshot(obj.reader);
            width = size(frame,2)/416;
            height = size(frame,1)/416;
            frame1 = imresize(im2single(frame),[416 416]);
            
            detections = detectObjects(frame1);
            
            % Run the tracker on the preprocessed detections.
            confirmedTracks = updateTracks(tracker, detections, frameCount);
            
            displayTrackingResults(obj, confirmedTracks, frame);
end


%% Create System Objects
% Create System objects used for reading the video frames

    function obj = setupSystemObjects()
        obj.reader = webcam('Logitech');
        obj.videoPlayer = vision.VideoPlayer;
        frame = snapshot(obj.reader);
        step(obj.videoPlayer, frame);
    end

%% Detect Objects
% The |detectObjects| function returns the centroids, labels and the 
% bounding boxes of the detected objects. Conditional if loop was set 
% to allow only display of 'person' labels.   
% 

    function detections = detectObjects(frame)
        
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
        
        measurementNoise = 100*eye(2); 
        
        % Formulate the detections as a list of objectDetection objects.
        numDetections = size(centroids, 1);
        detections = cell(numDetections, 1);
        for i = 1:numDetections
            detections{i} = objectDetection(frameCount, centroids(i,:), ...
                'MeasurementNoise', measurementNoise, ...
                'ObjectAttributes', {bboxes(i,:)});
        end
    end

%% Display Tracking Results
% The |displayTrackingResults| function draws a bounding box and label ID
% for each track on the video frame. 

    function displayTrackingResults(videoObjects, confirmedTracks, frame)
        % Convert the frame and the mask to uint8 RGB.
        frame = im2uint8(frame);
        
        if ~isempty(confirmedTracks)            
            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            numRelTr = numel(confirmedTracks);
            boxes = zeros(numRelTr, 4);
            ids = zeros(numRelTr, 1, 'int32');
            predictedTrackInds = zeros(numRelTr, 1);
            for tr = 1:numRelTr
                % Get bounding boxes.
                boxes(tr, :) = confirmedTracks(tr).ObjectAttributes{1}{1};
                
                % Get IDs.
                ids(tr) = confirmedTracks(tr).TrackID;
                
                if confirmedTracks(tr).IsCoasted
                    predictedTrackInds(tr) = tr;
                end
            end
            
            predictedTrackInds = predictedTrackInds(predictedTrackInds > 0);
            
            % Create labels for objects that display the predicted rather 
            % than the actual location.
            labels = cellstr(int2str(ids));
            
            isPredicted = cell(size(labels));
            isPredicted(predictedTrackInds) = {' predicted'};
            labels = strcat(labels, isPredicted);
            
            % Draw the objects on the frame.
            frame = insertObjectAnnotation(frame, 'rectangle', boxes, labels);

        end
        
        % Display the frame.       
        videoObjects.videoPlayer.step(frame);
    end

end
