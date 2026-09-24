# Engineering Notes

The NR-310 Vision Pod shares its calibration firmware with the NR-220. A regression in the
2025Q1 firmware release caused intermittent depth-estimation drift on units running in high
ambient infrared, which affected a small number of APAC deployments. The fix shipped in
firmware 4.2.1 and no NR-100 units were affected.

Manipulator margin is thinner than perception margin. Roughly speaking, the NR-100 carries a
lower gross margin than the NR-310 despite its much higher unit price, because the arm
assembly is largely bought-in while the Vision Pod is built in-house.

The Platform team owns the Fleet Controller, NR-450. It is the only product with a recurring
software component, and it is the strategic focus for 2026 because it pulls through
manipulator and perception hardware sales.
