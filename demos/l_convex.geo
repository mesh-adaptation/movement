h = 0.1;
Point(1) = {-1, -1, 0, h};
Point(2) = { 1, -1, 0, h};
Point(3) = { 1,  0, 0, h};
Point(4) = { 0,  0, 0, h};
Point(5) = { 0,  1, 0, h};
Point(6) = {-1,  1, 0, h};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
Line Loop(7) = {1, 2, 3, 4, 5, 6};
Plane Surface(7) = {7};

Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};
Physical Line(5) = {5};
Physical Line(6) = {6};
Physical Surface(7) = {7};

Point(7) = { 1,  1, 0, h};
Line(7) = {5, 7};
Line(8) = {7, 3};
Line Loop(8) = {3, 4, 7, 8};
Plane Surface(8) = {8};

Physical Line(7) = {7};
Physical Line(8) = {8};
Physical Surface(8) = {8};
