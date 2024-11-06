%% load data

close all
clear all
clc

load('');
firstThreshold = 15.0;
secondThreshold = 30.0;
height = 0.4;

%% synchronization and equilibrium computation

i = 1; 
j = 1;

while abs(roll(i)) < 1
    i = i + 1;
end
while abs(pitch(j)) < 1
    j = j + 1;
end

start = min(i,j);
dt = [0 : timeInterval : length(roll)*timeInterval-timeInterval];

roll = roll(start:end);
pitch = pitch(start:end);
dt = dt(start:end) - dt(start);

equilibriumPitch = mean(pitch(1:10/timeInterval));
equilibriumRoll = mean(roll(1:10/timeInterval));

%% plot Roll and Pitch angles in time separately

% Plot Roll with Thresholds
figure;
hRoll = plot(dt, roll, 'DisplayName', 'Roll Angle', 'LineWidth', 1); hold on;
xlabel('Time (s)', 'FontSize', 12);
ylabel('Angle (degrees)', 'FontSize', 12);
title('Roll Angle over Time', 'FontSize', 12);

hThresh1Roll = yline(15 + equilibriumRoll, '--', 'Color', [1, 0.5, 0], 'LineWidth', 1);
yline(-15 + equilibriumRoll, '--', 'Color', [1, 0.5, 0], 'LineWidth', 1);
hThresh2Roll = yline(30 + equilibriumRoll, 'r--', 'LineWidth', 1);
yline(-30 + equilibriumRoll, 'r--', 'LineWidth', 1);

legend([hRoll, hThresh1Roll, hThresh2Roll], {'Roll Angle', 'First Threshold', 'Second Threshold'}, 'FontSize', 12);
hold off;

% Plot Pitch with Thresholds
figure;
hRoll = plot(dt, pitch, 'DisplayName', 'Pitch Angle', 'LineWidth', 1); hold on;
xlabel('Time (s)', 'FontSize', 12);
ylabel('Angle (degrees)', 'FontSize', 12);
title('Pitch Angle over Time', 'FontSize', 12);

hThresh1Pitch = yline(15 + equilibriumRoll, '--', 'Color', [1, 0.5, 0], 'LineWidth', 1);
yline(-15 + equilibriumRoll, '--', 'Color', [1, 0.5, 0], 'LineWidth', 1);
hThresh2Pitch = yline(30 + equilibriumRoll, 'r--', 'LineWidth', 1);
yline(-30 + equilibriumRoll, 'r--', 'LineWidth', 1);

legend([hRoll, hThresh1Pitch, hThresh2Pitch], {'Pitch Angle', 'First Threshold', 'Second Threshold'}, 'FontSize', 12);
hold off;

%% 3D trajectory

eqAnglePitch = deg2rad(equilibriumPitch);
eqAngleRoll = deg2rad(equilibriumRoll);

px = zeros(length(roll), 1);
py = zeros(length(roll), 1);
pz = zeros(length(roll), 1);

for i = 1:length(roll)
    
    R = angle2dcm(deg2rad(roll(i)), deg2rad(pitch(i)), 0, 'XYZ');
    position = R * [0; 0; height];
    
    px(i) = position(1);
    py(i) = position(2);
    pz(i) = position(3);
end


% % Data filtering for smoother visualization
% windowSize = 51;
% polynomialOrder = 3;
% 
% px = sgolayfilt(px, polynomialOrder, windowSize);
% py = sgolayfilt(py, polynomialOrder, windowSize);
% pz = sgolayfilt(pz, polynomialOrder, windowSize);

figure;
hTraj = plot3(px, py, pz, 'DisplayName', 'Trajectory', 'LineWidth', 1);
hold on;
title('3D Trajectory', 'FontSize', 12);
xlabel('X Position (m)', 'FontSize', 12);
ylabel('Y Position (m)', 'FontSize', 12);
zlabel('Z Position (m)', 'FontSize', 12);
grid on;

firstDislocationAngle = deg2rad(firstThreshold);
secondDislocationAngle = deg2rad(secondThreshold);

[u1, v1] = meshgrid(linspace(0, 2 * pi, 100), linspace(0, firstDislocationAngle, 100));
x1 = height * cos(u1) .* sin(v1);
y1 = height * sin(u1) .* sin(v1);
z1 = height * ones(size(u1)) .* cos(v1);

rotationMatrix1 = angle2dcm(0, eqAnglePitch, 0, 'XYZ');

new_coords1 = rotationMatrix1 * [x1(:)'; y1(:)'; z1(:)'];
x3 = reshape(new_coords1(1, :), size(x1));
y3 = reshape(new_coords1(2, :), size(y1));
z3 = reshape(new_coords1(3, :), size(z1));

hSphere1 = surf(x3, y3, z3, 'FaceColor', [1, 0.5, 0], 'FaceAlpha', 0.3, 'EdgeColor', 'none'); % Orange color in RGB

[u2, v2] = meshgrid(linspace(0, 2 * pi, 100), linspace(0, secondDislocationAngle, 100));
x2 = height * cos(u2) .* sin(v2);
y2 = height * sin(u2) .* sin(v2);
z2 = height * ones(size(u2)) .* cos(v2);

rotationMatrix2 = angle2dcm(0, eqAnglePitch, 0, 'XYZ');

new_coords2 = rotationMatrix2 * [x2(:)'; y2(:)'; z2(:)'];
x4 = reshape(new_coords2(1, :), size(x2));
y4 = reshape(new_coords2(2, :), size(y2));
z4 = reshape(new_coords2(3, :), size(z2));

hSphere2 = surf(x4, y4, z4, 'FaceColor', [1, 0, 0], 'FaceAlpha', 0.3, 'EdgeColor', 'none'); % Red color in RGB

axis equal;

legend([hTraj, hSphere1, hSphere2], {'Trajectory', 'First Threshold', 'Second Threshold'}, 'FontSize', 12);
hold off;

%% data analysis

timeCount1 = 0.0;
timeBalanced1 = [];
timeCount2 = 0.0;
timeBalanced2 = [];
totTime = 0.0;

for cont = 1 : length(roll)
    if abs(roll(cont)-equilibriumRoll) < firstThreshold && abs(pitch(cont)-equilibriumPitch) < firstThreshold
        timeCount1 = timeCount1 + dt(cont);
    end
    if (abs(roll(cont)-equilibriumRoll) >= firstThreshold || abs(pitch(cont)-equilibriumPitch) >= firstThreshold) && timeCount1 ~= 0.0
        timeBalanced1 = [timeBalanced1 timeCount1];
        timeCount1 = 0;
    end
    
    if abs(pitch(cont)-equilibriumPitch) >= firstThreshold && timeCount1 ~= 0.0
        timeBalanced1 = [timeBalanced1 timeCount1];
        timeCount1 = 0;
    end
    
    if abs(roll(cont)-equilibriumRoll) < secondThreshold && abs(pitch(cont)-equilibriumPitch) < secondThreshold
        timeCount2 = timeCount2 + dt(cont);
    end
    if (abs(roll(cont)-equilibriumRoll) >= secondThreshold || abs(pitch(cont)-equilibriumPitch) >= secondThreshold) && timeCount2 ~= 0.0
        timeBalanced2 = [timeBalanced2 timeCount2];
        timeCount2 = 0;
    end
    totTime = totTime + dt(cont);
end

finish1 = 0; finish2 = 0;

if timeCount1 ~= 0
    timeBalanced1 = [timeBalanced1 timeCount1];
else
    finish1 = 1;
end
    
if timeCount2 ~= 0
    timeBalanced2 = [timeBalanced2 timeCount2];
else
    finish2 = 1;
end

fprintf('The session lasted %.1f seconds\n', totTime);
fprintf('The position has been maintained inside the %.0f threshold for as long as %.1f consecutive seconds\n', ...
    firstThreshold, max(timeBalanced1));
fprintf('The position has been maintained inside the %.0f threshold for as long as %.1f consecutive seconds\n', ...
    secondThreshold, max(timeBalanced2));
fprintf('The position has been maintained inside the %.0f threshold for a total of %.1f seconds, corresponding to %.0f%% of the whole session\n', ...
    firstThreshold, sum(timeBalanced1), sum(timeBalanced1)/totTime*100);
fprintf('The position has been maintained inside the %.0f threshold for a total of %.1f seconds, corresponding to %.0f%% of the whole session\n', ...
    secondThreshold, sum(timeBalanced2), sum(timeBalanced2)/totTime*100);
fprintf('The %.0f threshold has been exceeded %.0f times\n', firstThreshold, length(timeBalanced1)+finish1-1);
fprintf('The %.0f threshold has been exceeded %.0f times\n', secondThreshold, length(timeBalanced2)+finish2-1);

fprintf('Time intervals inside the %.0f threshold: [  ', firstThreshold);
for counter1 = 1:length(timeBalanced1)
    fprintf('%.1fs  ', timeBalanced1(counter1));
end
fprintf(']\n')

fprintf('Time intervals inside the %.0f threshold: [  ', secondThreshold);
for counter2 = 1:length(timeBalanced2)
    fprintf('%.1fs  ', timeBalanced2(counter2));
end
fprintf(']\n')
