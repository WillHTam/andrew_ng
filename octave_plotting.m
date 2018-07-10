% > must be done from cli apparently <

t = [0:0.01:0.98];
y1 = sin(2*pi*4*t);
plot(t,y1); 

y2= cos(2*pi*4*t);
plot(t,y2);

% to plot on top of each other
plot(t,y1);
hold on; % i want to plot something else on top
plot(t,y2,'r'); % 'r' is a color
xlabel('time')
ylabel('value')
legend('sin', 'cos')
title('my plot')
cd '~/Downloads'
print -dpng 'myPlot.png'
close % close plot

% open different windows
figure(1); plot(t, y1);
figure(2); plot(t, y2);

% subplotting
subplot(1,2,1); % divide figure into 1x2 grid, and access first element
plot(t, y1);
subplot(1,2,2);
plot(t, y2);
axis([0.5 1 -1 1])

clf; % clear figure

% visualize a matrix
A = magic(5)
imagesc(A), colorbar;
