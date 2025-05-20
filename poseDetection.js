/**
 * @file poseDetection.js
 * @description Standalone module for pose detection and interpretation from keypoints.
 * This module takes an array of keypoints (e.g., from MoveNet) and interprets them
 * into a defined set of pose states. It includes logic for smoothing keypoints,
 * detecting actions like punches, leans, and ducks, and managing internal state
 * related to pose analysis.
 *
 * Exports:
 *  - interpretPose(rawKeypoints, externalDebugUpdater): Function to process keypoints and return pose state.
 *  - POSE_STATES: Enum of possible pose states.
 *  - POSE_DETECTION_CONFIG: Default configuration object for pose detection parameters.
 *  - getSmoothedKeypoints(): Function to retrieve the latest smoothed keypoints for drawing.
 *  - getPoseDetectionInternalState(): Function to retrieve a copy of the internal state for debugging.
 *  - resetPoseDetectionState(): Function to reset the internal state.
 */

export const POSE_DETECTION_CONFIG = {
  minPoseConfidence: 0.3,
  poseSmoothFactor: 0.8,
  minTorsoHeight: 30,
  minShoulderWidth: 30,
  torsoChangeThreshold: 0.2,
  duckThresholdRatio: 0.35,
  leanThresholdX: 0.15,
  punchThresholdX: 0.3,
  temporarilyIncompleteDurationMs: 500,
  shoulderHipDistanceMultiplier: 0.5,
  standingHeightWindowSize: 30,
  minimumDuckDistance: 15,
  bodyScaleEMAlpha: 0.1,
  punchMinPixelSEDistance: 10,
  punchYAlignmentFactor: 0.6,
  minFramesForValidStandingHeight: 15,
  smoothingPasses: 1,
  minEssentialKeypointsForValidPose: 7, // Min # of essential keypoints to consider pose valid
};

export const POSE_STATES = {
  IDLE: 'IDLE',
  PUNCH_LEFT: 'PUNCH_LEFT',
  PUNCH_RIGHT: 'PUNCH_RIGHT',
  LEAN_LEFT: 'LEAN_LEFT',
  LEAN_RIGHT: 'LEAN_RIGHT',
  DUCK: 'DUCK',
  POSE_TEMPORARILY_INCOMPLETE: 'POSE_TEMPORARILY_INCOMPLETE',
  POSE_INCOMPLETE: 'POSE_INCOMPLETE',
  PUNCH_BOTH: 'PUNCH_BOTH',
  POSE_LOST: 'POSE_LOST'
};

const essentialNames = [
  "nose", "left_shoulder", "right_shoulder",
  "left_hip", "right_hip", "left_wrist",
  "right_wrist", "left_elbow", "right_elbow"
];

let internalState = {
  smoothedKeypoints: {},
  lastTorsoHeight: null,
  standingNoseY: null,
  recentNoseYs: [],
  ducking: false,
  temporarilyIncompleteEntryTime: 0,
  bodyScale: 1.0,
  standingTorsoHeight: null,
};

export function resetPoseDetectionState() {
    internalState = {
        smoothedKeypoints: {},
        lastTorsoHeight: null,
        standingNoseY: null,
        recentNoseYs: [],
        ducking: false,
        temporarilyIncompleteEntryTime: 0,
        bodyScale: 1.0,
        standingTorsoHeight: null,
    };
}


function _smoothAndGetKeypoint(name, map) {
  const kp = map[name];
  if(!kp) return null;

  let sk = internalState.smoothedKeypoints[name];
  if(!sk) {
    sk = {x: kp.x, y: kp.y, score: kp.score};
    internalState.smoothedKeypoints[name] = sk;
    return sk;
  }

  for (let i = 0; i < POSE_DETECTION_CONFIG.smoothingPasses; i++) {
      const f = POSE_DETECTION_CONFIG.poseSmoothFactor;
      sk.x = f * sk.x + (1-f) * kp.x;
      sk.y = f * sk.y + (1-f) * kp.y;
  }
  sk.score = kp.score;
  return sk;
}

function _checkPunch(w, e, s, isLeft, shoulderMidY, shoulderWidth) {
  if(!w || !e || !s) return false;

  const wristAhead = isLeft ? (w.x > e.x && e.x > s.x) : (w.x < e.x && e.x < s.x);
  if(!wristAhead) return false;

  const seDist = Math.abs(s.x - e.x),
        ewDist = Math.abs(e.x - w.x),
        alignedY = Math.abs(w.y - shoulderMidY) < shoulderWidth * POSE_DETECTION_CONFIG.punchYAlignmentFactor;

  return seDist > POSE_DETECTION_CONFIG.punchMinPixelSEDistance && ewDist > POSE_DETECTION_CONFIG.punchThresholdX * seDist && alignedY;
}

function _updateBodyScale(torsoHeight, shoulderWidth, debugUpdater) {
  const idealTorsoHeight = 150;
  const scale = torsoHeight / idealTorsoHeight;
  
  internalState.bodyScale = internalState.bodyScale * (1 - POSE_DETECTION_CONFIG.bodyScaleEMAlpha) + scale * POSE_DETECTION_CONFIG.bodyScaleEMAlpha;
  
  if (debugUpdater) debugUpdater('Body Scale', internalState.bodyScale);
  return internalState.bodyScale;
}

function _updateStandingHeight(nose, torsoHeight, debugUpdater) {
  if (!nose) return null;
  
  internalState.recentNoseYs.push(nose.y);
  
  if (internalState.recentNoseYs.length > POSE_DETECTION_CONFIG.standingHeightWindowSize) {
    internalState.recentNoseYs.shift();
  }
  
  if (internalState.recentNoseYs.length < POSE_DETECTION_CONFIG.minFramesForValidStandingHeight) {
    return null;
  }
  
  const sortedYs = [...internalState.recentNoseYs].sort((a, b) => a - b);
  const medianY = sortedYs[Math.floor(sortedYs.length / 2)];
  
  if (!internalState.standingNoseY || 
      Math.abs(medianY - internalState.standingNoseY) / Math.max(1, torsoHeight) > POSE_DETECTION_CONFIG.torsoChangeThreshold) {
    internalState.standingNoseY = medianY;
    internalState.standingTorsoHeight = torsoHeight;
    internalState.ducking = false;
  }
  
  return internalState.standingNoseY;
}

export function interpretPose(rawKeypoints, externalDebugUpdater = null) {
  if (!rawKeypoints || rawKeypoints.length === 0) {
    internalState.temporarilyIncompleteEntryTime = 0;
    return POSE_STATES.POSE_LOST;
  }

  const kpMap = {};
  let essentialKeypointsCount = 0;
  for (const p of rawKeypoints) {
      if (p && p.name && p.score >= POSE_DETECTION_CONFIG.minPoseConfidence) {
          kpMap[p.name] = p;
          if (essentialNames.includes(p.name)) {
            essentialKeypointsCount++;
          }
      }
  }

  const allEssentialPresent = essentialNames.every(n => kpMap[n]);
  const enoughEssentialPresent = essentialKeypointsCount >= POSE_DETECTION_CONFIG.minEssentialKeypointsForValidPose;

  if (!allEssentialPresent || !enoughEssentialPresent) {
    if (internalState.temporarilyIncompleteEntryTime === 0) {
      internalState.temporarilyIncompleteEntryTime = performance.now();
      return POSE_STATES.POSE_TEMPORARILY_INCOMPLETE;
    } else {
      if (performance.now() - internalState.temporarilyIncompleteEntryTime < POSE_DETECTION_CONFIG.temporarilyIncompleteDurationMs) {
        return POSE_STATES.POSE_TEMPORARILY_INCOMPLETE;
      } else {
        return POSE_STATES.POSE_INCOMPLETE;
      }
    }
  }

  internalState.temporarilyIncompleteEntryTime = 0;

  const es = {};
  for (let n of essentialNames) {
      if (kpMap[n]) { // Should always be true due to allEssentialPresent check
        es[n] = _smoothAndGetKeypoint(n, kpMap);
      } else { 
        return POSE_STATES.POSE_INCOMPLETE; // Should not happen if logic is correct
      }
  }

  const {
    nose, left_shoulder: ls, right_shoulder: rs,
    left_hip: lh, right_hip: rh,
    left_wrist: lw, right_wrist: rw,
    left_elbow: le, right_elbow: re
  } = es;

  // Check if all *smoothed* essential keypoints are valid (they could be null if smoothing failed or kpMap was sparse)
  const allSmoothedEssentialValid = essentialNames.every(name => es[name] && es[name].score >= POSE_DETECTION_CONFIG.minPoseConfidence);
  if (!allSmoothedEssentialValid) {
      return POSE_STATES.POSE_INCOMPLETE;
  }

  const shoulderMidY = (ls.y + rs.y) / 2,
        shoulderMidX = (ls.x + rs.x) / 2,
        hipMidY = (lh.y + rh.y) / 2,
        // hipMidX = (lh.x + rh.x) / 2, // hipMidX not used for torsoHeight, only for lean and duck's dynamic threshold
        torsoHeight = Math.abs(shoulderMidY - hipMidY),
        shoulderWidth = Math.abs(ls.x - rs.x);

  if (torsoHeight < POSE_DETECTION_CONFIG.minTorsoHeight || shoulderWidth < POSE_DETECTION_CONFIG.minShoulderWidth) {
    return POSE_STATES.POSE_LOST;
  }

  const bodyScale = _updateBodyScale(torsoHeight, shoulderWidth, externalDebugUpdater);
  const standingNoseY = _updateStandingHeight(nose, torsoHeight, externalDebugUpdater);
  
  internalState.lastTorsoHeight = torsoHeight;

  if (externalDebugUpdater) {
    externalDebugUpdater('Torso Height', torsoHeight);
    externalDebugUpdater('Shoulder Width', shoulderWidth);
    externalDebugUpdater('Standing Nose Y', standingNoseY);
    externalDebugUpdater('Current Nose Y', nose.y);
  }
  
  if (standingNoseY) {
    const dropPixels = nose.y - standingNoseY;
    const normalizedDrop = dropPixels / Math.max(1, torsoHeight); // Use current torsoHeight for normalization
    const dynamicThreshold = POSE_DETECTION_CONFIG.duckThresholdRatio;
    
    if (externalDebugUpdater) {
        externalDebugUpdater('Drop Pixels', dropPixels);
        externalDebugUpdater('Normalized Drop', normalizedDrop);
        externalDebugUpdater('Duck Threshold (Norm)', dynamicThreshold);
    }
    
    const minRequiredDrop = POSE_DETECTION_CONFIG.minimumDuckDistance * bodyScale;
    
    if (!internalState.ducking && dropPixels > minRequiredDrop && normalizedDrop > dynamicThreshold) {
        internalState.ducking = true;
    }
    else if (internalState.ducking && (dropPixels < minRequiredDrop * 0.5 || normalizedDrop < dynamicThreshold * 0.5)) {
        internalState.ducking = false;
    }
    if (externalDebugUpdater) externalDebugUpdater('Is Ducking', internalState.ducking);
  }

  if (internalState.ducking) {
      return POSE_STATES.DUCK;
  }

  const hipMidX = (lh.x + rh.x) / 2; // Calculate only if needed
  const leanOffset = shoulderMidX - hipMidX;
  const leanThresh = POSE_DETECTION_CONFIG.leanThresholdX * shoulderWidth;

  if (externalDebugUpdater) {
    externalDebugUpdater('Lean Offset', leanOffset);
    externalDebugUpdater('Lean Threshold', leanThresh);
  }

  if (leanOffset > leanThresh) return POSE_STATES.LEAN_LEFT;
  if (leanOffset < -leanThresh) return POSE_STATES.LEAN_RIGHT;

  const leftPunch = _checkPunch(lw, le, ls, true, shoulderMidY, shoulderWidth);
  const rightPunch = _checkPunch(rw, re, rs, false, shoulderMidY, shoulderWidth);

  if (externalDebugUpdater) {
    externalDebugUpdater('Left Punch Detected', leftPunch);
    externalDebugUpdater('Right Punch Detected', rightPunch);
  }

  if (leftPunch && rightPunch) return POSE_STATES.PUNCH_BOTH;
  if (leftPunch) return POSE_STATES.PUNCH_LEFT;
  if (rightPunch) return POSE_STATES.PUNCH_RIGHT;

  return POSE_STATES.IDLE;
}

export function getSmoothedKeypoints() {
    return internalState.smoothedKeypoints;
}

export function getPoseDetectionInternalState() {
    // Return a structured subset or a copy to prevent direct external modification
    return {
        ducking: internalState.ducking,
        standingNoseY: internalState.standingNoseY,
        bodyScale: internalState.bodyScale,
        // Add other relevant state parts if needed for external use (e.g. drawing)
    };
}
