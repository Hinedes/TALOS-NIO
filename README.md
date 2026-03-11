# Project TALOS: Neural-Inertial Odometry (NIO)

## Mission Objective
To engineer a heavily bounded, drift-resistant 6-DOF (Degrees of Freedom) spatial tracking foundation for an open-source mixed reality headset. This system relies purely on inertial data to maintain accurate spatial awareness before any visual camera data is introduced.

## The Core Problem: The Silicon Drift
Standard inertial measurement units (IMUs) are inherently noisy. When calculating position by integrating raw acceleration over time, microscopic thermal drifts and mechanical imperfections in the silicon compound rapidly compound. In traditional systems, this unobservable drift causes the headset's tracking to violently pull into the floor or ceiling within seconds. 

## The TALOS NIO Solution: A Dual-Loop Architecture
TALOS solves the drift problem by splitting the computational workload into two distinct, mathematically synchronized engines:

* **The Kinematic Engine (The Fast Loop):** A continuous physics filter that ingests raw sensor data and integrates it instantly to maintain the ultra-low latency tracking required for spatial computing.
* **The Neural Engine (The Slow Loop):** A lightweight machine learning model that analyzes a rolling window of the sensor's frequency spectrum. Instead of learning generic physics, it acts as a dedicated hardware profiler. It recognizes the specific biomechanical signatures of human movement and outputs a clean velocity prediction.
* **The Fusion Gate (Rewind and Replay):** Because the neural network takes time to execute, its predictions arrive slightly delayed. The TALOS system briefly rewinds its state history, compares the neural network's velocity guess against its own physics math, and uses the difference to calculate the exact hardware bias. It actively strips this thermal and mechanical noise out of the raw data before fast-forwarding back to the present millisecond.

## The Ultimate Goal
By locking the unobservable Z-axis drift into a tightly controlled spatial envelope, this NIO pipeline serves as the unshakeable bedrock for the headset. It guarantees that when the optical tracking layer (Visual-Neural-Inertial Odometry) is eventually integrated, the cameras will only have to map the room, rather than fighting the basic physics of gravity.
