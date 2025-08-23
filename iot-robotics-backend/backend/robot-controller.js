/**
 * Robot Control Module
 * Implements kinematics, trajectory planning, and real-time control
 */

class RobotController {
    constructor(robotId) {
        this.robotId = robotId;
        
        // DH Parameters for 6-DOF robot (in meters)
        this.dhParams = [
            { theta: 0, d: 0.290, a: 0.000, alpha: Math.PI/2 },   // Joint 1
            { theta: 0, d: 0.000, a: 0.270, alpha: 0 },           // Joint 2
            { theta: 0, d: 0.000, a: 0.070, alpha: Math.PI/2 },   // Joint 3
            { theta: 0, d: 0.302, a: 0.000, alpha: -Math.PI/2 },  // Joint 4
            { theta: 0, d: 0.000, a: 0.000, alpha: Math.PI/2 },   // Joint 5
            { theta: 0, d: 0.072, a: 0.000, alpha: 0 }            // Joint 6
        ];
        
        // Joint limits (radians)
        this.jointLimits = [
            { min: -Math.PI, max: Math.PI },        // Joint 1: ±180°
            { min: -Math.PI/2, max: Math.PI/2 },    // Joint 2: ±90°
            { min: -Math.PI, max: Math.PI },        // Joint 3: ±180°
            { min: -2*Math.PI, max: 2*Math.PI },    // Joint 4: ±360°
            { min: -Math.PI/2, max: Math.PI/2 },    // Joint 5: ±90°
            { min: -2*Math.PI, max: 2*Math.PI }     // Joint 6: ±360°
        ];
        
        // Velocity and acceleration limits
        this.maxVelocity = [3.14, 3.14, 3.14, 6.28, 6.28, 6.28]; // rad/s
        this.maxAcceleration = [10, 10, 10, 20, 20, 20]; // rad/s²
        this.maxJerk = [50, 50, 50, 100, 100, 100]; // rad/s³
        
        // Current state
        this.currentJointPositions = [0, 0, 0, 0, 0, 0];
        this.currentJointVelocities = [0, 0, 0, 0, 0, 0];
        this.currentCartesianPosition = null;
        this.currentCartesianOrientation = null;
        
        // Trajectory execution
        this.activeTrajectory = null;
        this.trajectoryStartTime = null;
        this.isExecuting = false;
        
        // Safety
        this.emergencyStop = false;
        this.collisionThreshold = 0.05; // meters
        
        // Update current cartesian position
        this.updateCartesianState();
    }
    
    /**
     * DH Transformation Matrix
     */
    dhTransform(theta, d, a, alpha) {
        const ct = Math.cos(theta);
        const st = Math.sin(theta);
        const ca = Math.cos(alpha);
        const sa = Math.sin(alpha);
        
        return [
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ];
    }
    
    /**
     * Matrix multiplication for 4x4 matrices
     */
    matrixMultiply(A, B) {
        const result = Array(4).fill().map(() => Array(4).fill(0));
        
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                for (let k = 0; k < 4; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return result;
    }
    
    /**
     * Forward Kinematics - Calculate end-effector position from joint angles
     */
    forwardKinematics(jointAngles) {
        let T = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ];
        
        for (let i = 0; i < 6; i++) {
            const theta = jointAngles[i] + this.dhParams[i].theta;
            const d = this.dhParams[i].d;
            const a = this.dhParams[i].a;
            const alpha = this.dhParams[i].alpha;
            
            const Ti = this.dhTransform(theta, d, a, alpha);
            T = this.matrixMultiply(T, Ti);
        }
        
        // Extract position
        const position = [T[0][3], T[1][3], T[2][3]];
        
        // Extract orientation (Euler angles ZYX)
        const r11 = T[0][0], r12 = T[0][1], r13 = T[0][2];
        const r21 = T[1][0], r22 = T[1][1], r23 = T[1][2];
        const r31 = T[2][0], r32 = T[2][1], r33 = T[2][2];
        
        const sy = Math.sqrt(r11*r11 + r21*r21);
        const singular = sy < 1e-6;
        
        let x, y, z;
        if (!singular) {
            x = Math.atan2(r32, r33);
            y = Math.atan2(-r31, sy);
            z = Math.atan2(r21, r11);
        } else {
            x = Math.atan2(-r23, r22);
            y = Math.atan2(-r31, sy);
            z = 0;
        }
        
        return {
            position: position,
            orientation: [x, y, z],
            transform: T
        };
    }
    
    /**
     * Inverse Kinematics - Calculate joint angles from target position
     * Simplified analytical solution for 6-DOF robot
     */
    inverseKinematics(targetPosition, targetOrientation) {
        // This is a simplified IK solution
        // In production, use a proper analytical or numerical IK solver
        
        const px = targetPosition[0];
        const py = targetPosition[1];
        const pz = targetPosition[2];
        
        // Wrist position (accounting for tool offset)
        const wx = px - 0.072 * Math.cos(targetOrientation[2]);
        const wy = py - 0.072 * Math.sin(targetOrientation[2]);
        const wz = pz;
        
        // Joint 1
        const theta1 = Math.atan2(wy, wx);
        
        // Joint 2 and 3 (using geometric approach)
        const r = Math.sqrt(wx*wx + wy*wy);
        const s = wz - 0.290;
        const D = (r*r + s*s - 0.270*0.270 - 0.372*0.372) / (2 * 0.270 * 0.372);
        
        if (Math.abs(D) > 1) {
            throw new Error('Target position unreachable');
        }
        
        const theta3 = Math.atan2(Math.sqrt(1 - D*D), D);
        const theta2 = Math.atan2(s, r) - Math.atan2(0.372 * Math.sin(theta3), 
                                                      0.270 + 0.372 * Math.cos(theta3));
        
        // Joints 4-6 (orientation)
        // Simplified - in production, calculate from rotation matrix
        const theta4 = targetOrientation[0];
        const theta5 = targetOrientation[1];
        const theta6 = targetOrientation[2];
        
        return [theta1, theta2, theta3, theta4, theta5, theta6];
    }
    
    /**
     * Quintic Polynomial Trajectory
     */
    quinticTrajectory(q0, qf, duration) {
        // 5th order polynomial with zero velocity/acceleration at endpoints
        const a0 = q0;
        const a1 = 0;
        const a2 = 0;
        const a3 = 10 * (qf - q0) / Math.pow(duration, 3);
        const a4 = -15 * (qf - q0) / Math.pow(duration, 4);
        const a5 = 6 * (qf - q0) / Math.pow(duration, 5);
        
        return {
            position: (t) => {
                return a0 + a1*t + a2*t*t + a3*t*t*t + a4*t*t*t*t + a5*t*t*t*t*t;
            },
            velocity: (t) => {
                return a1 + 2*a2*t + 3*a3*t*t + 4*a4*t*t*t + 5*a5*t*t*t*t;
            },
            acceleration: (t) => {
                return 2*a2 + 6*a3*t + 12*a4*t*t + 20*a5*t*t*t;
            }
        };
    }
    
    /**
     * Trapezoidal Velocity Profile
     */
    trapezoidalTrajectory(q0, qf, vMax, aMax) {
        const distance = Math.abs(qf - q0);
        const direction = Math.sign(qf - q0);
        
        // Calculate profile times
        const tAcc = vMax / aMax;
        const dAcc = 0.5 * aMax * tAcc * tAcc;
        
        let tConst, tTotal;
        
        if (2 * dAcc > distance) {
            // Triangle profile
            tAcc = Math.sqrt(distance / aMax);
            tConst = 0;
            tTotal = 2 * tAcc;
        } else {
            // Trapezoidal profile
            const dConst = distance - 2 * dAcc;
            tConst = dConst / vMax;
            tTotal = 2 * tAcc + tConst;
        }
        
        return {
            duration: tTotal,
            position: (t) => {
                if (t <= tAcc) {
                    // Acceleration phase
                    return q0 + direction * 0.5 * aMax * t * t;
                } else if (t <= tAcc + tConst) {
                    // Constant velocity phase
                    const dt = t - tAcc;
                    return q0 + direction * (dAcc + vMax * dt);
                } else if (t <= tTotal) {
                    // Deceleration phase
                    const dt = t - tAcc - tConst;
                    const dDecel = vMax * dt - 0.5 * aMax * dt * dt;
                    return q0 + direction * (dAcc + vMax * tConst + dDecel);
                } else {
                    return qf;
                }
            },
            velocity: (t) => {
                if (t <= tAcc) {
                    return direction * aMax * t;
                } else if (t <= tAcc + tConst) {
                    return direction * vMax;
                } else if (t <= tTotal) {
                    const dt = tTotal - t;
                    return direction * aMax * dt;
                } else {
                    return 0;
                }
            }
        };
    }
    
    /**
     * Plan trajectory for multiple joints
     */
    planTrajectory(targetJoints, trajectoryType = 'quintic', velocityScale = 1.0) {
        const waypoints = [];
        const duration = this.calculateDuration(targetJoints, velocityScale);
        const numSamples = Math.ceil(duration * 100); // 100Hz sampling
        
        const trajectories = [];
        
        for (let i = 0; i < 6; i++) {
            const q0 = this.currentJointPositions[i];
            const qf = targetJoints[i];
            
            if (trajectoryType === 'quintic') {
                trajectories.push(this.quinticTrajectory(q0, qf, duration));
            } else if (trajectoryType === 'trapezoidal') {
                const vMax = this.maxVelocity[i] * velocityScale;
                const aMax = this.maxAcceleration[i] * velocityScale;
                trajectories.push(this.trapezoidalTrajectory(q0, qf, vMax, aMax));
            }
        }
        
        // Generate waypoints
        for (let sample = 0; sample <= numSamples; sample++) {
            const t = (sample / numSamples) * duration;
            const jointPositions = [];
            const jointVelocities = [];
            
            for (let i = 0; i < 6; i++) {
                jointPositions.push(trajectories[i].position(t));
                jointVelocities.push(trajectories[i].velocity(t));
            }
            
            const fk = this.forwardKinematics(jointPositions);
            
            waypoints.push({
                time: t,
                jointPositions: jointPositions,
                jointVelocities: jointVelocities,
                cartesianPosition: fk.position,
                cartesianOrientation: fk.orientation
            });
        }
        
        return {
            trajectoryId: `traj_${Date.now()}`,
            duration: duration,
            waypoints: waypoints,
            type: trajectoryType
        };
    }
    
    /**
     * Calculate trajectory duration based on joint limits
     */
    calculateDuration(targetJoints, velocityScale) {
        let maxDuration = 0;
        
        for (let i = 0; i < 6; i++) {
            const distance = Math.abs(targetJoints[i] - this.currentJointPositions[i]);
            const vMax = this.maxVelocity[i] * velocityScale;
            const aMax = this.maxAcceleration[i] * velocityScale;
            
            // Time for trapezoidal profile
            const tAcc = vMax / aMax;
            const dAcc = 0.5 * aMax * tAcc * tAcc;
            
            let duration;
            if (2 * dAcc > distance) {
                // Triangle profile
                duration = 2 * Math.sqrt(distance / aMax);
            } else {
                // Trapezoidal profile
                const dConst = distance - 2 * dAcc;
                const tConst = dConst / vMax;
                duration = 2 * tAcc + tConst;
            }
            
            maxDuration = Math.max(maxDuration, duration);
        }
        
        return maxDuration;
    }
    
    /**
     * Execute trajectory
     */
    async executeTrajectory(trajectory) {
        if (this.isExecuting) {
            throw new Error('Already executing a trajectory');
        }
        
        this.activeTrajectory = trajectory;
        this.trajectoryStartTime = Date.now();
        this.isExecuting = true;
        
        return new Promise((resolve, reject) => {
            const executionInterval = setInterval(() => {
                if (this.emergencyStop) {
                    clearInterval(executionInterval);
                    this.isExecuting = false;
                    reject(new Error('Emergency stop activated'));
                    return;
                }
                
                const elapsed = (Date.now() - this.trajectoryStartTime) / 1000;
                
                if (elapsed >= trajectory.duration) {
                    // Trajectory complete
                    const finalWaypoint = trajectory.waypoints[trajectory.waypoints.length - 1];
                    this.currentJointPositions = finalWaypoint.jointPositions;
                    this.currentJointVelocities = [0, 0, 0, 0, 0, 0];
                    this.updateCartesianState();
                    
                    clearInterval(executionInterval);
                    this.isExecuting = false;
                    
                    resolve({
                        success: true,
                        finalPosition: this.currentCartesianPosition,
                        finalJoints: this.currentJointPositions,
                        executionTime: elapsed
                    });
                } else {
                    // Interpolate current position
                    const waypointIndex = Math.floor((elapsed / trajectory.duration) * trajectory.waypoints.length);
                    const waypoint = trajectory.waypoints[Math.min(waypointIndex, trajectory.waypoints.length - 1)];
                    
                    this.currentJointPositions = waypoint.jointPositions;
                    this.currentJointVelocities = waypoint.jointVelocities;
                    this.currentCartesianPosition = waypoint.cartesianPosition;
                    this.currentCartesianOrientation = waypoint.cartesianOrientation;
                }
            }, 10); // 100Hz control loop
        });
    }
    
    /**
     * Move to target position
     */
    async moveTo(target, options = {}) {
        const {
            motionType = 'joint',
            velocityScale = 0.5,
            accelerationScale = 0.5,
            trajectoryType = 'quintic'
        } = options;
        
        let targetJoints;
        
        if (motionType === 'joint') {
            targetJoints = target.joints || target;
        } else if (motionType === 'cartesian') {
            // Convert cartesian target to joint angles
            targetJoints = this.inverseKinematics(target.position, target.orientation);
        }
        
        // Validate joint limits
        for (let i = 0; i < 6; i++) {
            if (targetJoints[i] < this.jointLimits[i].min || targetJoints[i] > this.jointLimits[i].max) {
                throw new Error(`Joint ${i+1} exceeds limits: ${targetJoints[i]} rad`);
            }
        }
        
        // Plan trajectory
        const trajectory = this.planTrajectory(targetJoints, trajectoryType, velocityScale);
        
        // Execute trajectory
        return await this.executeTrajectory(trajectory);
    }
    
    /**
     * Jog robot in specified direction
     */
    jog(axis, direction, speed = 0.1) {
        if (this.isExecuting) {
            return { success: false, error: 'Already executing motion' };
        }
        
        const jogIncrement = direction * speed * 0.01; // Small increment
        
        if (axis < 6) {
            // Joint jog
            const newJoints = [...this.currentJointPositions];
            newJoints[axis] += jogIncrement;
            
            // Check limits
            if (newJoints[axis] >= this.jointLimits[axis].min && 
                newJoints[axis] <= this.jointLimits[axis].max) {
                this.currentJointPositions[axis] = newJoints[axis];
                this.updateCartesianState();
                return { success: true, position: this.currentJointPositions };
            }
        } else {
            // Cartesian jog
            const cartAxis = axis - 6; // 6=X, 7=Y, 8=Z
            const newPosition = [...this.currentCartesianPosition];
            newPosition[cartAxis] += jogIncrement;
            
            try {
                const newJoints = this.inverseKinematics(newPosition, this.currentCartesianOrientation);
                this.currentJointPositions = newJoints;
                this.updateCartesianState();
                return { success: true, position: this.currentCartesianPosition };
            } catch (error) {
                return { success: false, error: 'Position unreachable' };
            }
        }
        
        return { success: false, error: 'Limit reached' };
    }
    
    /**
     * Update cartesian state from joint positions
     */
    updateCartesianState() {
        const fk = this.forwardKinematics(this.currentJointPositions);
        this.currentCartesianPosition = fk.position;
        this.currentCartesianOrientation = fk.orientation;
    }
    
    /**
     * Home robot
     */
    async home() {
        const homePosition = [0, 0, 0, 0, 0, 0];
        return await this.moveTo(homePosition, {
            motionType: 'joint',
            velocityScale: 0.3,
            trajectoryType: 'trapezoidal'
        });
    }
    
    /**
     * Emergency stop
     */
    stop() {
        this.emergencyStop = true;
        this.isExecuting = false;
        this.currentJointVelocities = [0, 0, 0, 0, 0, 0];
        return { success: true, stopped: true };
    }
    
    /**
     * Reset emergency stop
     */
    reset() {
        this.emergencyStop = false;
        return { success: true, reset: true };
    }
    
    /**
     * Get current state
     */
    getState() {
        return {
            robotId: this.robotId,
            jointPositions: this.currentJointPositions,
            jointVelocities: this.currentJointVelocities,
            cartesianPosition: this.currentCartesianPosition,
            cartesianOrientation: this.currentCartesianOrientation,
            isExecuting: this.isExecuting,
            emergencyStop: this.emergencyStop
        };
    }
    
    /**
     * Execute program (sequence of moves)
     */
    async executeProgram(program) {
        const results = [];
        
        for (const step of program) {
            if (this.emergencyStop) {
                break;
            }
            
            try {
                const result = await this.moveTo(step.target, step.options);
                results.push({ step: step.name, success: true, result });
                
                // Add delay if specified
                if (step.delay) {
                    await new Promise(resolve => setTimeout(resolve, step.delay));
                }
            } catch (error) {
                results.push({ step: step.name, success: false, error: error.message });
                break;
            }
        }
        
        return results;
    }
}

module.exports = RobotController;