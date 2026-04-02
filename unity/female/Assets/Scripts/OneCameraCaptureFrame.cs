// OneCameraCaptureFrame.cs
// Single-view capture: one camera, no orbit/rotation.
// - Freeze animator at sampled normalizedTime
// - Render via cam.Render() into RenderTexture
// - Save PNG + 2D keypoints per sampled frame
//
// Put this file under Assets/Scripts/

using System;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace PoseRecord.Data
{
    [Serializable]
    public class Joint2DData
    {
        public string name;
        public float x, y, conf; // viewport 0~1 (Top-left if flipYToTopLeft=true)
    }

    [Serializable]
    public class Frame2DRecordData
    {
        public int frame_idx;
        public float time;
        public string camera;
        public int width;
        public int height;
        public float angle_deg; // not used (0)
        public string label;
        public List<Joint2DData> joints2d = new List<Joint2DData>();
    }

    [Serializable]
    public class SequenceMetaData
    {
        public string subject_id;
        public string action_id;
        public string camera_id;
        public int total_frames;
        public int sampled_frames;
        public int joints_count;
        public int pose_every_n_frames;
        public int width;
        public int height;
        public float fov;
        public string created_at_utc;
    }

    [Serializable]
    public class CameraIntrinsicsData
    {
        public int width;
        public int height;
        public float fx;
        public float fy;
        public float cx;
        public float cy;
    }

    [Serializable]
    public class CameraExtrinsicsData
    {
        public float[] t_world_cam_4x4;
        public float[] position_world_xyz;
        public float[] rotation_world_quat_xyzw;
    }

    [Serializable]
    public class DatasetManifestData
    {
        public string updated_at_utc;
        public List<DatasetActionEntry> actions = new List<DatasetActionEntry>();
    }

    [Serializable]
    public class DatasetActionEntry
    {
        public string subject_id;
        public string action_id;
        public string action_path;
        public int camera_count;
        public string[] camera_ids;
    }

    [Serializable]
    public class JointNamesData
    {
        public string[] joint_names;
    }
}

public class OneCameraCaptureFrame : MonoBehaviour
{
    static readonly string[] PoleJointNames = new string[]
    {
        "Pole_L_Handle", "Pole_L_Tip", "Pole_R_Handle", "Pole_R_Tip"
    };

    static readonly string[] SkiJointNames = new string[]
    {
        "Ski_L_center", "Ski_L_front", "Ski_L_back", "Ski_R_center", "Ski_R_front", "Ski_R_back"
    };

    static class GlobalFrameSync
    {
        public static int expectedParticipants = 1;
        public static int registeredParticipants = 0;
        public static int currentStep = 0;
        public static int arrivedAtStep = 0;
        public static int generation = 0;

        public static void Configure(int participants)
        {
            expectedParticipants = Mathf.Max(1, participants);
            registeredParticipants = 0;
            currentStep = 0;
            arrivedAtStep = 0;
            generation = 0;
        }

        public static void Register()
        {
            registeredParticipants++;
        }

        public static void Unregister()
        {
            registeredParticipants = Mathf.Max(0, registeredParticipants - 1);
            if (registeredParticipants == 0)
            {
                currentStep = 0;
                arrivedAtStep = 0;
                generation = 0;
            }
        }

        public static void Arrive(int step)
        {
            if (step != currentStep) return;
            arrivedAtStep++;
            if (arrivedAtStep >= expectedParticipants)
            {
                arrivedAtStep = 0;
                currentStep++;
                generation++;
            }
        }
    }

    public static void ConfigureGlobalSyncParticipants(int participants)
    {
        GlobalFrameSync.Configure(participants);
        Debug.Log($"[OneCameraCaptureFrame] Global frame sync configured: participants={Mathf.Max(1, participants)}");
    }

    struct ClipSegment
    {
        public AnimationClip clip;
        public int startFrame;
        public int frameCount;
    }

    class ActionCaptureState
    {
        public string actionName;
        public string actionRoot;
        public string metaFolder;
        public string cameraFolder;
        public string imageFolder;
        public string vizFolder;
        public string vizCharacterFolder;
        public string vizPoleFolder;
        public string vizSkiFolder;
        public string kpt2dFolder;
        public string kpt2dCharacterPath;
        public string kpt2dPolePath;
        public string kpt2dSkiPath;
        public string kpt2dCharacterFolder;
        public string kpt2dPoleFolder;
        public string kpt2dSkiFolder;
        public List<float> kpt2dCharacterBuffer;
        public List<float> kpt2dPoleBuffer;
        public List<float> kpt2dSkiBuffer;
        public int characterJointCount;
        public int poleJointCount;
        public int skiJointCount;
        public int totalFrames;
        public int sampledFrames;
    }

    [Header("Output")]
    public string outRootFolder = "SkiDataset";
    public string subjectId = "S001";
    public string actionId = "A001";
    public string cameraId = "";

    [Tooltip("人物目录名（位于 SkiDataset 与 动作目录 之间）。为空时可自动使用 target 名称")]
    public string characterFolderName = "";

    [Tooltip("characterFolderName 为空时，自动使用 target.name 作为人物目录名")]
    public bool autoUseTargetNameAsCharacterFolder = false;

    [Tooltip("使用动作名（首个采样 clip 名）作为 actions 下的目录名")]
    public bool useClipNameAsActionFolder = true;

    [Header("Output Naming")]
    [Tooltip("帧目录名前缀，例如 capture_L0_A000")]
    public string captureFolderPrefix = "capture";
    [Tooltip("图片文件名前缀，例如 frame_000001.png")]
    public string imagePrefix = "frame";
    [Tooltip("2D 关键点 npy 文件名前缀，例如 kpt2d.npy")]
    public string kptPrefix = "kpt2d";

    [Tooltip("可视化图片文件名前缀（仅用于可视化输出目录）")]
    public string vizImagePrefix = "viz";

    [Tooltip("相机内外参只保存一次到 SkiDataset/cameras/<cameraId>/")]
    public bool saveCameraMetaOnlyOnce = true;

    [Header("Auto Run")]
    public bool autoRunOnPlay = true;
    public float startDelaySec = 0.0f;

    [Header("Target")]
    public Transform target;            // 人物 root / pelvis
    public Animator targetAnimator;     // Animator（可选）

    [Tooltip("每隔多少个 Unity 帧采一次姿态")]
    public int poseEveryNFrames = 1;

    [Tooltip("强制每帧采样（忽略 poseEveryNFrames，推荐开启）")]
    public bool forceEveryFrame = true;

    [Tooltip("Animator layer index")]
    public int animatorLayer = 0;

    [Header("Animation Sampling")]
    [Tooltip("按 Controller 内全部动作顺序采样并计算 total_frames（而不是只采当前动作）")]
    public bool sampleAllControllerClips = true;

    [Tooltip("打印 Controller 动作明细与总长度")]
    public bool logControllerClipSummary = true;

    [Tooltip("按动作（clip）拆分输出目录（建议关闭：frames/kpt2d 仅按相机分）")]
    public bool splitOutputByClip = false;

    [Header("Joints")]
    [Tooltip("是否自动扫描 rootBone 下的骨骼")]
    public bool autoScanAllChildren = true;

    public Transform rootBone;
    public Transform[] joints;

    [Header("Camera / Capture")]
    public Camera cam;
    public int captureWidth = 1920;
    public int captureHeight = 1080;

    [Header("Keypoint Coordinate")]
    public bool flipYToTopLeft = true;
    public bool flipX = false;

    [Header("Camera Preset")]
    public bool applyInspectorLikePreset = true;
    public float presetFov = 70f;
    public float presetNear = 0.3f;
    public float presetFar = 1000f;
    public int presetTargetDisplay = 0; // 建议 0
    public int presetDepth = -1;

    [Tooltip("是否在数据集根目录生成/刷新 dataset_manifest.json")]
    public bool exportDatasetManifest = true;

    [Tooltip("在每个动作的 meta 目录导出 joint_names.json（用于可视化筛选）")]
    public bool exportJointNamesMeta = true;

    [Header("Cross-Camera Frame Sync")]
    [Tooltip("开启后：所有相机在每个采样 step 上同步，保证同一帧采样后再进入下一帧")]
    public bool enableGlobalFrameSync = true;

    [Tooltip("等待所有相机注册的最大秒数（超时则继续，避免死等）")]
    public float globalSyncWaitTimeoutSec = 10f;

    [Header("NPZ Export")]
    [Tooltip("同时导出 2D 关键点 npz（与 npy 并存）")]
    public bool exportKpt2dNpz = true;

    // 新增：每帧单独导出 kpt2d npy
    [Tooltip("每帧单独导出 2D 关键点 npy 到 kpt2d 文件夹")]
    public bool exportKpt2dPerFrame = true;

    [Header("Visualization Export")]
    [Tooltip("同时导出每帧可视化图片（将 kpt2d 叠加在 frame 上）")]
    public bool exportKpt2dOverlayImage = true;

    [Tooltip("可视化关键点半径（像素）")]
    public int vizPointRadius = 3;

    [Header("Log Export")]
    [Tooltip("将采样/保存过程日志同时写入文件")]
    public bool saveOutputLogToFile = true;

    [Tooltip("日志文件名前缀")]
    public string outputLogFilePrefix = "capture_log";

    // session
    private float baseTime;

    // resources
    private RenderTexture rt;
    private Texture2D tex;
    private StreamWriter captureLogWriter;
    private string captureLogPath;

    public bool IsCaptureDone { get; private set; } = false;
    public bool IsCaptureStarted { get; private set; } = false;

    IEnumerator Start()
    {
        IsCaptureStarted = true;
        IsCaptureDone = false;
        bool syncRegistered = false;
        try
        {
            if (!autoRunOnPlay) yield break;

            if (cam == null) cam = GetComponent<Camera>();
            if (cam == null)
            {
                Debug.LogError("[OneCameraCaptureFrame] Camera is null.");
                yield break;
            }

            if (string.IsNullOrWhiteSpace(cameraId))
                cameraId = cam.name;

            if (enableGlobalFrameSync)
            {
                GlobalFrameSync.Register();
                syncRegistered = true;
                yield return StartCoroutine(WaitForGlobalSyncReady());
            }

            if (target == null)
            {
                Debug.LogError("[OneCameraCaptureFrame] target is null.");
                yield break;
            }

            if (applyInspectorLikePreset)
                ApplyCameraPresetLikeInspector();

            // resolve joints
            if (autoScanAllChildren && rootBone != null)
            {
                var list = new List<Transform>();
                var smr = rootBone.GetComponentInChildren<SkinnedMeshRenderer>();
                if (smr != null && smr.bones != null && smr.bones.Length > 0)
                    list.AddRange(smr.bones);
                else
                {
                    list.Add(rootBone);
                    GetAllChildren(rootBone, list);
                }
                joints = list.ToArray();
            }

            if (joints == null || joints.Length == 0)
            {
                Debug.LogError("[OneCameraCaptureFrame] joints is empty.");
                yield break;
            }

            if (startDelaySec > 0f)
                yield return new WaitForSeconds(startDelaySec);

            if (targetAnimator == null) targetAnimator = target.GetComponentInChildren<Animator>();
            if (targetAnimator == null)
            {
                Debug.LogError("[OneCameraCaptureFrame] targetAnimator is null.");
                yield break;
            }

            RuntimeAnimatorController ac = targetAnimator.runtimeAnimatorController;
            if (ac == null)
            {
                Debug.LogError("[OneCameraCaptureFrame] targetAnimator.runtimeAnimatorController is null.");
                yield break;
            }

            targetAnimator.enabled = true;
            targetAnimator.Update(0f);

            AnimationClip captureClip = null;
            var currentClips = targetAnimator.GetCurrentAnimatorClipInfo(animatorLayer);
            if (currentClips != null && currentClips.Length > 0)
                captureClip = currentClips[0].clip;
            if (captureClip == null && ac.animationClips != null && ac.animationClips.Length > 0)
                captureClip = ac.animationClips[0];

            if (captureClip == null)
            {
                Debug.LogError("[OneCameraCaptureFrame] no valid AnimationClip found for sampling.");
                yield break;
            }

            var controllerClips = GetUniqueControllerClips(ac);
            var clipSegments = BuildClipSegments(sampleAllControllerClips ? controllerClips : new List<AnimationClip> { captureClip });

            int totalFrames = GetTotalFramesFromSegments(clipSegments);
            if (logControllerClipSummary)
                LogClipSegmentsSummary(clipSegments, sampleAllControllerClips ? "ControllerAllClips" : "CurrentClipOnly");

            if (totalFrames <= 0)
            {
                Debug.LogError("[OneCameraCaptureFrame] totalFrames <= 0，无法采样。");
                yield break;
            }

            string resolvedCharacterFolder = ResolveCharacterFolderName();

            baseTime = Time.time;
            yield return StartCoroutine(CaptureSequence(totalFrames, clipSegments, resolvedCharacterFolder));

            IsCaptureDone = true;
        }
        finally
        {
            CloseCaptureLog();
            if (syncRegistered)
                GlobalFrameSync.Unregister();
        }
    }

    IEnumerator CaptureSequence(int totalFrames, List<ClipSegment> clipSegments, string characterFolder)
    {
        string root = Path.Combine(Application.dataPath, "..", "..", outRootFolder);
        string characterRoot = Path.Combine(root, characterFolder);
        InitCaptureLog(characterRoot);
        string safeCameraIdForKpt2d = string.IsNullOrWhiteSpace(cameraId) ? "cam_unknown" : cameraId;
        string captureFolderName = string.IsNullOrWhiteSpace(captureFolderPrefix)
            ? cameraId
            : $"{captureFolderPrefix}_{cameraId}";
        string kpt2dFileName = string.IsNullOrWhiteSpace(kptPrefix) ? "kpt2d.npy" : $"{kptPrefix}.npy";
        LogInfo($"[OneCameraCaptureFrame] Capture started. character={characterFolder}, camera={cameraId}, root={characterRoot}");

        if (saveCameraMetaOnlyOnce)
        {
            string sharedCameraFolder = Path.Combine(characterRoot, "cameras", safeCameraIdForKpt2d);
            Directory.CreateDirectory(sharedCameraFolder);
            string intrPath = Path.Combine(sharedCameraFolder, "intrinsics.json");
            string extrPath = Path.Combine(sharedCameraFolder, "extrinsics.json");
            if (!File.Exists(intrPath)) WriteCameraIntrinsics(intrPath);
            if (!File.Exists(extrPath)) WriteCameraExtrinsics(extrPath);
        }

        var actionTotalFrames = new Dictionary<string, int>();
        for (int i = 0; i < clipSegments.Count; i++)
        {
            var seg = clipSegments[i];
            string segName = (seg.clip != null && !string.IsNullOrWhiteSpace(seg.clip.name)) ? seg.clip.name : actionId;
            string safeSegName = MakeSafePathName(segName);
            if (!actionTotalFrames.ContainsKey(safeSegName)) actionTotalFrames[safeSegName] = 0;
            actionTotalFrames[safeSegName] += seg.frameCount;
        }

        // allocate RT & texture
        rt = new RenderTexture(captureWidth, captureHeight, 24, RenderTextureFormat.ARGB32);
        rt.Create();
        tex = new Texture2D(captureWidth, captureHeight, TextureFormat.RGBA32, false);

        // Keep camera projection and render target geometry aligned with the capture RT.
        cam.targetTexture = rt;
        cam.rect = new Rect(0f, 0f, 1f, 1f);
        cam.aspect = rt.width / (float)rt.height;
        cam.ResetProjectionMatrix();
        cam.targetTexture = null;

        int stride = forceEveryFrame ? 1 : Mathf.Max(1, poseEveryNFrames);

        // record template
        var rec = new PoseRecord.Data.Frame2DRecordData();
        rec.camera = cam.name;
        rec.joints2d = new List<PoseRecord.Data.Joint2DData>(joints.Length);
        foreach (var j in joints)
            rec.joints2d.Add(new PoseRecord.Data.Joint2DData { name = j.name });

        float oldSpeed = targetAnimator ? targetAnimator.speed : 1f;
        bool oldEnabled = targetAnimator ? targetAnimator.enabled : false;

        int expectedSamples = Mathf.CeilToInt(totalFrames / (float)stride);
        if ((totalFrames - 1) % stride != 0) expectedSamples += 1;
        int[] characterIndices = BuildAllJointIndices();
        int[] poleIndices = FindJointIndicesByExactNames(PoleJointNames);
        int[] skiIndices = FindJointIndicesByExactNames(SkiJointNames);

        if (poleIndices.Length != PoleJointNames.Length)
            LogWarning($"[OneCameraCaptureFrame] Pole subset resolved {poleIndices.Length}/{PoleJointNames.Length}. Check pole joint names.");
        if (skiIndices.Length != SkiJointNames.Length)
            LogWarning($"[OneCameraCaptureFrame] Ski subset resolved {skiIndices.Length}/{SkiJointNames.Length}. Check ski joint names.");

        var actionStates = new Dictionary<string, ActionCaptureState>();

        ActionCaptureState EnsureActionState(string safeActionName)
        {
            if (actionStates.TryGetValue(safeActionName, out var existing)) return existing;

            string actionRoot = Path.Combine(characterRoot, safeActionName);
            string metaFolder = Path.Combine(actionRoot, "meta");
            string cameraFolder = Path.Combine(actionRoot, "cameras", cameraId);
            string imageFolder = Path.Combine(actionRoot, "frames", captureFolderName);
            string vizFolder = Path.Combine(actionRoot, "viz", captureFolderName);
            string vizCharacterFolder = Path.Combine(vizFolder, "character");
            string vizPoleFolder = Path.Combine(vizFolder, "pole");
            string vizSkiFolder = Path.Combine(vizFolder, "ski");
            string kpt2dFolder = Path.Combine(actionRoot, "kpt2d", safeCameraIdForKpt2d);
            string kpt2dCharacterFolder = Path.Combine(kpt2dFolder, "character");
            string kpt2dPoleFolder = Path.Combine(kpt2dFolder, "pole");
            string kpt2dSkiFolder = Path.Combine(kpt2dFolder, "ski");
            string kpt2dCharacterPath = Path.Combine(kpt2dCharacterFolder, kpt2dFileName);
            string kpt2dPolePath = Path.Combine(kpt2dPoleFolder, kpt2dFileName);
            string kpt2dSkiPath = Path.Combine(kpt2dSkiFolder, kpt2dFileName);

            Directory.CreateDirectory(metaFolder);
            Directory.CreateDirectory(imageFolder);
            if (exportKpt2dOverlayImage)
            {
                Directory.CreateDirectory(vizCharacterFolder);
                Directory.CreateDirectory(vizPoleFolder);
                Directory.CreateDirectory(vizSkiFolder);
            }
            Directory.CreateDirectory(kpt2dCharacterFolder);
            Directory.CreateDirectory(kpt2dPoleFolder);
            Directory.CreateDirectory(kpt2dSkiFolder);

            if (!saveCameraMetaOnlyOnce)
            {
                Directory.CreateDirectory(cameraFolder);
                WriteCameraIntrinsics(Path.Combine(cameraFolder, "intrinsics.json"));
                WriteCameraExtrinsics(Path.Combine(cameraFolder, "extrinsics.json"));
            }

            var state = new ActionCaptureState
            {
                actionName = safeActionName,
                actionRoot = actionRoot,
                metaFolder = metaFolder,
                cameraFolder = cameraFolder,
                imageFolder = imageFolder,
                vizFolder = vizFolder,
                vizCharacterFolder = vizCharacterFolder,
                vizPoleFolder = vizPoleFolder,
                vizSkiFolder = vizSkiFolder,
                kpt2dFolder = kpt2dFolder,
                kpt2dCharacterPath = kpt2dCharacterPath,
                kpt2dPolePath = kpt2dPolePath,
                kpt2dSkiPath = kpt2dSkiPath,
                kpt2dCharacterFolder = kpt2dCharacterFolder,
                kpt2dPoleFolder = kpt2dPoleFolder,
                kpt2dSkiFolder = kpt2dSkiFolder,
                kpt2dCharacterBuffer = new List<float>(expectedSamples * Mathf.Max(1, characterIndices.Length) * 3),
                kpt2dPoleBuffer = new List<float>(expectedSamples * Mathf.Max(1, poleIndices.Length) * 3),
                kpt2dSkiBuffer = new List<float>(expectedSamples * Mathf.Max(1, skiIndices.Length) * 3),
                characterJointCount = characterIndices.Length,
                poleJointCount = poleIndices.Length,
                skiJointCount = skiIndices.Length,
                totalFrames = actionTotalFrames.TryGetValue(safeActionName, out var tf) ? tf : 0,
                sampledFrames = 0,
            };

            WriteSequenceMeta(Path.Combine(metaFolder, "sequence.json"), safeActionName, state.totalFrames, 0, stride);
            if (exportJointNamesMeta)
                WriteJointNamesMetaBySubset(metaFolder, characterIndices, poleIndices, skiIndices);
            actionStates.Add(safeActionName, state);

            LogInfo($"[OneCameraCaptureFrame] Output camera folder: {imageFolder}");
            return state;
        }

        int outputFrameIdx = 0;
        int globalStep = 0;

        for (int sampleIdx = 0; sampleIdx < totalFrames; sampleIdx += stride)
        {
            if (enableGlobalFrameSync)
                yield return StartCoroutine(WaitForGlobalStep(globalStep));

            if (targetAnimator && clipSegments != null && clipSegments.Count > 0)
                FreezeAnimatorAtSample(clipSegments, sampleIdx, totalFrames);

            int localFrame;
            var seg = GetSegmentAtFrame(clipSegments, sampleIdx, totalFrames, out localFrame);
            string actionName = (seg.clip != null && !string.IsNullOrWhiteSpace(seg.clip.name)) ? seg.clip.name : actionId;
            string safeActionName = MakeSafePathName(actionName);
            var state = EnsureActionState(safeActionName);

            // 传入 imagePrefix
            yield return StartCoroutine(CaptureSingleFrame(
                sampleIdx,
                state.sampledFrames,
                state.imageFolder,
                state.vizCharacterFolder,
                state.vizPoleFolder,
                state.vizSkiFolder,
                imagePrefix,
                kptPrefix,
                rec,
                state.kpt2dCharacterFolder,
                state.kpt2dPoleFolder,
                state.kpt2dSkiFolder,
                state.kpt2dCharacterBuffer,
                state.kpt2dPoleBuffer,
                state.kpt2dSkiBuffer,
                characterIndices,
                poleIndices,
                skiIndices
            ));

            state.sampledFrames++;

            outputFrameIdx++;

            if (enableGlobalFrameSync)
            {
                yield return StartCoroutine(ArriveAndWaitGlobalStep(globalStep));
                globalStep++;
            }
        }

        int lastFrameIdx = Mathf.Max(0, totalFrames - 1);
        bool lastFrameAlreadyCaptured = (lastFrameIdx % stride == 0);
        if (!lastFrameAlreadyCaptured)
        {
            if (enableGlobalFrameSync)
                yield return StartCoroutine(WaitForGlobalStep(globalStep));

            if (targetAnimator && clipSegments != null && clipSegments.Count > 0)
                FreezeAnimatorAtSample(clipSegments, lastFrameIdx, totalFrames);

            int localFrame;
            var seg = GetSegmentAtFrame(clipSegments, lastFrameIdx, totalFrames, out localFrame);
            string actionName = (seg.clip != null && !string.IsNullOrWhiteSpace(seg.clip.name)) ? seg.clip.name : actionId;
            string safeActionName = MakeSafePathName(actionName);
            var state = EnsureActionState(safeActionName);

            yield return StartCoroutine(CaptureSingleFrame(
                lastFrameIdx,
                state.sampledFrames,
                state.imageFolder,
                state.vizCharacterFolder,
                state.vizPoleFolder,
                state.vizSkiFolder,
                imagePrefix,
                kptPrefix,
                rec,
                state.kpt2dCharacterFolder,
                state.kpt2dPoleFolder,
                state.kpt2dSkiFolder,
                state.kpt2dCharacterBuffer,
                state.kpt2dPoleBuffer,
                state.kpt2dSkiBuffer,
                characterIndices,
                poleIndices,
                skiIndices
            ));

            state.sampledFrames++;

            outputFrameIdx++;

            if (enableGlobalFrameSync)
            {
                yield return StartCoroutine(ArriveAndWaitGlobalStep(globalStep));
                globalStep++;
            }
        }

        foreach (var kv in actionStates)
        {
            var state = kv.Value;
            if (state.characterJointCount > 0)
                WriteKpt2DNpy(state.kpt2dCharacterPath, state.kpt2dCharacterBuffer, state.sampledFrames, state.characterJointCount);
            if (state.poleJointCount > 0)
                WriteKpt2DNpy(state.kpt2dPolePath, state.kpt2dPoleBuffer, state.sampledFrames, state.poleJointCount);
            if (state.skiJointCount > 0)
                WriteKpt2DNpy(state.kpt2dSkiPath, state.kpt2dSkiBuffer, state.sampledFrames, state.skiJointCount);

            WriteSequenceMeta(Path.Combine(state.metaFolder, "sequence.json"), state.actionName, state.totalFrames, state.sampledFrames, stride);
        }

        if (exportDatasetManifest)
            WriteDatasetManifest(characterRoot, characterFolder);

        LogInfo($"[OneCameraCaptureFrame] Capture summary | totalFrames={totalFrames}, stride={stride}, savedFrames={outputFrameIdx}, actions={actionStates.Count}");


        if (targetAnimator)
        {
            targetAnimator.speed = oldSpeed;
            targetAnimator.enabled = oldEnabled;
        }

        rt.Release();
        Destroy(rt);
        Destroy(tex);

        LogInfo("[OneCameraCaptureFrame] Done.");
    }

    IEnumerator WaitForGlobalSyncReady()
    {
        float begin = Time.realtimeSinceStartup;
        while (GlobalFrameSync.registeredParticipants < GlobalFrameSync.expectedParticipants)
        {
            if (globalSyncWaitTimeoutSec > 0f && Time.realtimeSinceStartup - begin > globalSyncWaitTimeoutSec)
            {
                LogWarning($"[OneCameraCaptureFrame] Global sync ready timeout. registered={GlobalFrameSync.registeredParticipants}, expected={GlobalFrameSync.expectedParticipants}");
                yield break;
            }
            yield return null;
        }
    }

    IEnumerator WaitForGlobalStep(int step)
    {
        while (GlobalFrameSync.currentStep < step)
            yield return null;
    }

    IEnumerator ArriveAndWaitGlobalStep(int step)
    {
        int gen = GlobalFrameSync.generation;
        GlobalFrameSync.Arrive(step);
        while (GlobalFrameSync.generation == gen)
            yield return null;
    }

    void FreezeAnimatorAtSample(List<ClipSegment> clipSegments, int sampleIdx, int totalFrames)
    {
        if (clipSegments == null || clipSegments.Count == 0 || targetAnimator == null) return;
        int localFrame;
        var seg = GetSegmentAtFrame(clipSegments, sampleIdx, totalFrames, out localFrame);

        float localDenom = Mathf.Max(1f, seg.frameCount - 1f);
        float localT01 = Mathf.Clamp01(localFrame / localDenom);
        float tSec = localT01 * seg.clip.length;

        targetAnimator.enabled = false;
        seg.clip.SampleAnimation(targetAnimator.gameObject, tSec);

        LogInfo($"采样帧: {sampleIdx}/{Mathf.Max(0, totalFrames - 1)} | 动作: {seg.clip.name} | 局部帧: {localFrame}/{Mathf.Max(0, seg.frameCount - 1)} | 时间: {tSec:F3}s/{seg.clip.length:F3}s");
    }

    ClipSegment GetSegmentAtFrame(List<ClipSegment> clipSegments, int sampleIdx, int totalFrames, out int localFrame)
    {
        int clampedFrame = Mathf.Clamp(sampleIdx, 0, Mathf.Max(0, totalFrames - 1));
        ClipSegment seg = clipSegments[clipSegments.Count - 1];
        localFrame = Mathf.Max(0, seg.frameCount - 1);

        for (int i = 0; i < clipSegments.Count; i++)
        {
            var s = clipSegments[i];
            int segEnd = s.startFrame + s.frameCount;
            if (clampedFrame < segEnd)
            {
                seg = s;
                localFrame = clampedFrame - s.startFrame;
                break;
            }
        }

        return seg;
    }

    string MakeSafePathName(string name)
    {
        string safe = string.IsNullOrWhiteSpace(name) ? "UnknownAction" : name;
        char[] invalid = Path.GetInvalidFileNameChars();
        for (int i = 0; i < invalid.Length; i++)
            safe = safe.Replace(invalid[i], '_');
        return safe;
    }

    List<AnimationClip> GetUniqueControllerClips(RuntimeAnimatorController ac)
    {
        var result = new List<AnimationClip>();
        if (ac == null || ac.animationClips == null) return result;

        var seen = new HashSet<AnimationClip>();
        foreach (var clip in ac.animationClips)
        {
            if (clip == null) continue;
            if (seen.Add(clip)) result.Add(clip);
        }
        return result;
    }

    List<ClipSegment> BuildClipSegments(List<AnimationClip> clips)
    {
        var segments = new List<ClipSegment>();
        if (clips == null || clips.Count == 0) return segments;

        int start = 0;
        foreach (var clip in clips)
        {
            if (clip == null) continue;
            int frames = Mathf.Max(1, Mathf.RoundToInt(clip.length * clip.frameRate));
            segments.Add(new ClipSegment
            {
                clip = clip,
                startFrame = start,
                frameCount = frames
            });
            start += frames;
        }
        return segments;
    }

    int GetTotalFramesFromSegments(List<ClipSegment> segments)
    {
        if (segments == null || segments.Count == 0) return 0;
        var last = segments[segments.Count - 1];
        return last.startFrame + last.frameCount;
    }

    void LogClipSegmentsSummary(List<ClipSegment> segments, string tag)
    {
        if (segments == null || segments.Count == 0)
        {
            LogWarning($"[OneCameraCaptureFrame] {tag}: no clip segments");
            return;
        }

        float totalSec = 0f;
        for (int i = 0; i < segments.Count; i++)
        {
            var s = segments[i];
            totalSec += s.clip.length;
            LogInfo($"[OneCameraCaptureFrame] {tag} clip[{i}] = {s.clip.name}, length={s.clip.length:F3}s, fps={s.clip.frameRate:F2}, frames={s.frameCount}");
        }

        int totalFrames = GetTotalFramesFromSegments(segments);
        LogInfo($"<color=yellow>[OneCameraCaptureFrame] {tag} summary: clips={segments.Count}, totalLength={totalSec:F3}s, totalFrames={totalFrames}</color>");
    }

    IEnumerator CaptureSingleFrame(
        int frameIdx,
        int outputFrameIdx,
        string imageFolder,
        string vizCharacterFolder,
        string vizPoleFolder,
        string vizSkiFolder,
        string imgPrefix,
        string kptPrefixName,
        PoseRecord.Data.Frame2DRecordData rec,
        string kpt2dCharacterFolder,
        string kpt2dPoleFolder,
        string kpt2dSkiFolder,
        List<float> kpt2dCharacterBuffer,
        List<float> kpt2dPoleBuffer,
        List<float> kpt2dSkiBuffer,
        int[] characterIndices,
        int[] poleIndices,
        int[] skiIndices
    )
    {
        string safeImagePrefix = string.IsNullOrWhiteSpace(imgPrefix) ? "frame" : imgPrefix;
        Directory.CreateDirectory(imageFolder);
        string imgPath = Path.Combine(imageFolder, $"{safeImagePrefix}_{outputFrameIdx:000000}.png");

        if (captureWidth <= 0 || captureHeight <= 0)
        {
            LogError($"[OneCameraCaptureFrame] Invalid capture size: {captureWidth}x{captureHeight}");
            yield break;
        }

        if (rt == null || !rt.IsCreated())
        {
            LogError("[OneCameraCaptureFrame] RenderTexture is null or not created.");
            yield break;
        }

        cam.targetTexture = rt;
        cam.Render();

        try
        {
            int w = rt.width;
            int h = rt.height;
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            tex.ReadPixels(new Rect(0, 0, w, h), 0, 0, false);
            tex.Apply(false, false);
            RenderTexture.active = prev;

            File.WriteAllBytes(imgPath, tex.EncodeToPNG());
        }
        catch (Exception ex)
        {
            LogError($"[OneCameraCaptureFrame] Failed to save PNG: {imgPath}\n{ex}");
        }
        finally
        {
            cam.targetTexture = null;
        }

        if (!File.Exists(imgPath))
            LogError($"[OneCameraCaptureFrame] PNG write failed: {imgPath}");
        else if (outputFrameIdx == 0)
            LogInfo($"[OneCameraCaptureFrame] First frame saved: {imgPath}");

        FillRecord(rec, frameIdx);

        float[] characterFrame = GetKeypointsAsPixelByIndices(characterIndices);
        float[] poleFrame = GetKeypointsAsPixelByIndices(poleIndices);
        float[] skiFrame = GetKeypointsAsPixelByIndices(skiIndices);

        AppendFrameToBuffer(kpt2dCharacterBuffer, characterFrame);
        AppendFrameToBuffer(kpt2dPoleBuffer, poleFrame);
        AppendFrameToBuffer(kpt2dSkiBuffer, skiFrame);

        if (exportKpt2dPerFrame)
        {
            string safeKptPrefix = string.IsNullOrWhiteSpace(kptPrefixName) ? "kpt2d" : kptPrefixName;
            if (characterIndices != null && characterIndices.Length > 0)
            {
                string perFramePath = Path.Combine(kpt2dCharacterFolder, $"{safeKptPrefix}_{outputFrameIdx:000000}.npy");
                WriteNpyFloat32_2D(perFramePath, characterFrame, characterIndices.Length, 3);
            }
            if (poleIndices != null && poleIndices.Length > 0)
            {
                string perFramePath = Path.Combine(kpt2dPoleFolder, $"{safeKptPrefix}_{outputFrameIdx:000000}.npy");
                WriteNpyFloat32_2D(perFramePath, poleFrame, poleIndices.Length, 3);
            }
            if (skiIndices != null && skiIndices.Length > 0)
            {
                string perFramePath = Path.Combine(kpt2dSkiFolder, $"{safeKptPrefix}_{outputFrameIdx:000000}.npy");
                WriteNpyFloat32_2D(perFramePath, skiFrame, skiIndices.Length, 3);
            }

            if (exportKpt2dOverlayImage)
            {
                string safeVizPrefix = string.IsNullOrWhiteSpace(vizImagePrefix) ? "viz" : vizImagePrefix;
                if (characterIndices != null && characterIndices.Length > 0)
                {
                    string vizPath = Path.Combine(vizCharacterFolder, $"{safeVizPrefix}_{outputFrameIdx:000000}.png");
                    SaveOverlayVisualization(vizPath, characterFrame, characterIndices.Length);
                }
                if (poleIndices != null && poleIndices.Length > 0)
                {
                    string vizPath = Path.Combine(vizPoleFolder, $"{safeVizPrefix}_{outputFrameIdx:000000}.png");
                    SaveOverlayVisualization(vizPath, poleFrame, poleIndices.Length);
                }
                if (skiIndices != null && skiIndices.Length > 0)
                {
                    string vizPath = Path.Combine(vizSkiFolder, $"{safeVizPrefix}_{outputFrameIdx:000000}.png");
                    SaveOverlayVisualization(vizPath, skiFrame, skiIndices.Length);
                }
            }
        }
        else if (exportKpt2dOverlayImage)
        {
            string safeVizPrefix = string.IsNullOrWhiteSpace(vizImagePrefix) ? "viz" : vizImagePrefix;
            if (characterIndices != null && characterIndices.Length > 0)
            {
                string vizPath = Path.Combine(vizCharacterFolder, $"{safeVizPrefix}_{outputFrameIdx:000000}.png");
                SaveOverlayVisualization(vizPath, characterFrame, characterIndices.Length);
            }
            if (poleIndices != null && poleIndices.Length > 0)
            {
                string vizPath = Path.Combine(vizPoleFolder, $"{safeVizPrefix}_{outputFrameIdx:000000}.png");
                SaveOverlayVisualization(vizPath, poleFrame, poleIndices.Length);
            }
            if (skiIndices != null && skiIndices.Length > 0)
            {
                string vizPath = Path.Combine(vizSkiFolder, $"{safeVizPrefix}_{outputFrameIdx:000000}.png");
                SaveOverlayVisualization(vizPath, skiFrame, skiIndices.Length);
            }
        }
    }

    void SaveOverlayVisualization(string path, float[] frameKpt, int jointCount)
    {
        if (tex == null || frameKpt == null || jointCount <= 0) return;

        int w = rt != null ? rt.width : captureWidth;
        int h = rt != null ? rt.height : captureHeight;

        Color32[] src = tex.GetPixels32();
        Color32[] dst = new Color32[src.Length];
        Array.Copy(src, dst, src.Length);

        int radius = Mathf.Max(1, vizPointRadius);
        for (int i = 0; i < jointCount; i++)
        {
            int baseIdx = i * 3;
            float x = frameKpt[baseIdx + 0];
            float y = frameKpt[baseIdx + 1];
            float c = frameKpt[baseIdx + 2];
            if (c <= 0f) continue;

            int px = Mathf.RoundToInt(x);
            int py = Mathf.RoundToInt(y);
            // frameKpt uses top-left origin when flipYToTopLeft=true, but Texture2D pixels are bottom-left origin.
            if (flipYToTopLeft)
                py = (h - 1) - py;

            DrawCircle(dst, w, h, px, py, radius, new Color32(0, 255, 102, 255));
        }

        var vizTex = new Texture2D(w, h, TextureFormat.RGBA32, false);
        vizTex.SetPixels32(dst);
        vizTex.Apply(false, false);

        try
        {
            File.WriteAllBytes(path, vizTex.EncodeToPNG());
        }
        catch (Exception ex)
        {
            LogError($"[OneCameraCaptureFrame] Failed to save viz PNG: {path}\n{ex}");
        }
        finally
        {
            Destroy(vizTex);
        }
    }

    void InitCaptureLog(string characterRoot)
    {
        if (!saveOutputLogToFile) return;

        try
        {
            string logsFolder = Path.Combine(characterRoot, "logs");
            Directory.CreateDirectory(logsFolder);

            string safeCam = MakeSafePathName(string.IsNullOrWhiteSpace(cameraId) ? "cam_unknown" : cameraId);
            string safePrefix = string.IsNullOrWhiteSpace(outputLogFilePrefix) ? "capture_log" : MakeSafePathName(outputLogFilePrefix);
            string ts = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            captureLogPath = Path.Combine(logsFolder, $"{safePrefix}_{safeCam}_{ts}.log");

            captureLogWriter = new StreamWriter(captureLogPath, false, Encoding.UTF8);
            captureLogWriter.AutoFlush = true;
            captureLogWriter.WriteLine($"[{DateTime.UtcNow:O}] [INFO] [OneCameraCaptureFrame] Log file created: {captureLogPath}");
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[OneCameraCaptureFrame] Failed to initialize output log file. {ex.Message}");
            captureLogWriter = null;
            captureLogPath = null;
        }
    }

    void CloseCaptureLog()
    {
        if (captureLogWriter == null) return;

        try
        {
            captureLogWriter.WriteLine($"[{DateTime.UtcNow:O}] [INFO] [OneCameraCaptureFrame] Log file closed.");
            captureLogWriter.Flush();
            captureLogWriter.Dispose();
        }
        catch
        {
        }
        finally
        {
            captureLogWriter = null;
            captureLogPath = null;
        }
    }

    void LogInfo(string msg)
    {
        Debug.Log(msg);
        WriteLogFileLine("INFO", msg);
    }

    void LogWarning(string msg)
    {
        Debug.LogWarning(msg);
        WriteLogFileLine("WARN", msg);
    }

    void LogError(string msg)
    {
        Debug.LogError(msg);
        WriteLogFileLine("ERROR", msg);
    }

    void WriteLogFileLine(string level, string msg)
    {
        if (!saveOutputLogToFile || captureLogWriter == null) return;

        try
        {
            captureLogWriter.WriteLine($"[{DateTime.UtcNow:O}] [{level}] {msg}");
        }
        catch
        {
        }
    }

    static void DrawCircle(Color32[] pixels, int width, int height, int cx, int cy, int radius, Color32 color)
    {
        int rr = radius * radius;
        int xMin = Mathf.Max(0, cx - radius);
        int xMax = Mathf.Min(width - 1, cx + radius);
        int yMin = Mathf.Max(0, cy - radius);
        int yMax = Mathf.Min(height - 1, cy + radius);

        for (int y = yMin; y <= yMax; y++)
        {
            int dy = y - cy;
            for (int x = xMin; x <= xMax; x++)
            {
                int dx = x - cx;
                if (dx * dx + dy * dy > rr) continue;
                pixels[y * width + x] = color;
            }
        }
    }

    int[] BuildAllJointIndices()
    {
        int n = joints != null ? joints.Length : 0;
        int[] indices = new int[n];
        for (int i = 0; i < n; i++)
            indices[i] = i;
        return indices;
    }

    int[] FindJointIndicesByExactNames(string[] names)
    {
        if (joints == null || names == null || names.Length == 0)
            return new int[0];

        var indexByName = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < joints.Length; i++)
        {
            var j = joints[i];
            if (j != null && !indexByName.ContainsKey(j.name))
                indexByName[j.name] = i;
        }

        var found = new List<int>(names.Length);
        for (int i = 0; i < names.Length; i++)
        {
            if (indexByName.TryGetValue(names[i], out int idx))
                found.Add(idx);
        }

        return found.ToArray();
    }

    float[] GetKeypointsAsPixelByIndices(int[] indices)
    {
        int w = rt != null ? rt.width : captureWidth;
        int h = rt != null ? rt.height : captureHeight;
        if (indices == null || indices.Length == 0)
            return new float[0];

        float[] frame = new float[indices.Length * 3];
        int o = 0;

        for (int i = 0; i < indices.Length; i++)
        {
            int jointIdx = indices[i];
            if (jointIdx < 0 || jointIdx >= joints.Length || joints[jointIdx] == null)
            {
                frame[o++] = 0f;
                frame[o++] = 0f;
                frame[o++] = 0f;
                continue;
            }

            Vector3 vp = cam.WorldToViewportPoint(joints[jointIdx].position);
            float conf = (vp.z > 0 && vp.x >= 0 && vp.x <= 1 && vp.y >= 0 && vp.y <= 1) ? 1f : 0f;

            float x01 = flipX ? 1f - vp.x : vp.x;
            float y01 = flipYToTopLeft ? 1f - vp.y : vp.y;

            float x = x01 * (w - 1);
            float y = y01 * (h - 1);

            frame[o++] = x;
            frame[o++] = y;
            frame[o++] = conf;
        }

        return frame;
    }

    void AppendFrameToBuffer(List<float> outBuffer, float[] frame)
    {
        if (outBuffer == null || frame == null || frame.Length == 0)
            return;
        for (int i = 0; i < frame.Length; i++)
            outBuffer.Add(frame[i]);
    }

    void WriteSequenceMeta(string path, string actionName, int totalFrames, int sampledFrames, int stride)
    {
        var seq = new PoseRecord.Data.SequenceMetaData
        {
            subject_id = subjectId,
            action_id = actionName,
            camera_id = cameraId,
            total_frames = totalFrames,
            sampled_frames = sampledFrames,
            joints_count = joints != null ? joints.Length : 0,
            pose_every_n_frames = Mathf.Max(1, stride),
            width = captureWidth,
            height = captureHeight,
            fov = cam != null ? cam.fieldOfView : presetFov,
            created_at_utc = DateTime.UtcNow.ToString("o")
        };

        File.WriteAllText(path, JsonUtility.ToJson(seq, true), Encoding.UTF8);
    }

    void WriteJointNamesMetaBySubset(string metaFolder, int[] characterIndices, int[] poleIndices, int[] skiIndices)
    {
        WriteJointNamesMeta(Path.Combine(metaFolder, "joint_names_character.json"), characterIndices);
        WriteJointNamesMeta(Path.Combine(metaFolder, "joint_names_pole.json"), poleIndices);
        WriteJointNamesMeta(Path.Combine(metaFolder, "joint_names_ski.json"), skiIndices);

        // Keep legacy filename for compatibility with existing tools.
        WriteJointNamesMeta(Path.Combine(metaFolder, "joint_names.json"), characterIndices);
    }

    void WriteJointNamesMeta(string path, int[] indices)
    {
        var names = ResolveJointNamesByIndices(indices);
        if (names == null || names.Length == 0) return;

        var data = new PoseRecord.Data.JointNamesData
        {
            joint_names = names
        };

        File.WriteAllText(path, JsonUtility.ToJson(data, true), Encoding.UTF8);
    }

    string[] ResolveJointNamesByIndices(int[] indices)
    {
        if (joints == null || joints.Length == 0 || indices == null || indices.Length == 0)
            return new string[0];

        var names = new string[indices.Length];
        for (int i = 0; i < indices.Length; i++)
        {
            int idx = indices[i];
            if (idx >= 0 && idx < joints.Length && joints[idx] != null)
                names[i] = joints[idx].name;
            else
                names[i] = $"joint_{idx}";
        }

        return names;
    }

    string ResolveActionFolderName(AnimationClip captureClip)
    {
        if (useClipNameAsActionFolder && captureClip != null && !string.IsNullOrWhiteSpace(captureClip.name))
            return MakeSafePathName(captureClip.name);

        if (!string.IsNullOrWhiteSpace(actionId))
            return MakeSafePathName(actionId);

        return "ActionUnknown";
    }

    string ResolveCharacterFolderName()
    {
        if (!string.IsNullOrWhiteSpace(subjectId))
            return MakeSafePathName(subjectId);

        return "CharacterUnknown";
    }

    void WriteCameraIntrinsics(string path)
    {
        float fovRad = cam.fieldOfView * Mathf.Deg2Rad;
        float fy = 0.5f * captureHeight / Mathf.Tan(0.5f * fovRad);
        float fx = fy * (captureWidth / (float)captureHeight);

        var intr = new PoseRecord.Data.CameraIntrinsicsData
        {
            width = captureWidth,
            height = captureHeight,
            fx = fx,
            fy = fy,
            cx = (captureWidth - 1) * 0.5f,
            cy = (captureHeight - 1) * 0.5f
        };

        File.WriteAllText(path, JsonUtility.ToJson(intr, true), Encoding.UTF8);
    }

    void WriteCameraExtrinsics(string path)
    {
        Matrix4x4 w2c = cam.worldToCameraMatrix;
        Quaternion q = cam.transform.rotation;
        Vector3 p = cam.transform.position;

        var ex = new PoseRecord.Data.CameraExtrinsicsData
        {
            t_world_cam_4x4 = new float[16]
            {
                w2c.m00, w2c.m01, w2c.m02, w2c.m03,
                w2c.m10, w2c.m11, w2c.m12, w2c.m13,
                w2c.m20, w2c.m21, w2c.m22, w2c.m23,
                w2c.m30, w2c.m31, w2c.m32, w2c.m33,
            },
            position_world_xyz = new float[] { p.x, p.y, p.z },
            rotation_world_quat_xyzw = new float[] { q.x, q.y, q.z, q.w }
        };

        File.WriteAllText(path, JsonUtility.ToJson(ex, true), Encoding.UTF8);
    }

    void WriteKpt2DNpy(string path, List<float> data, int t, int j)
    {
        using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write))
        using (var bw = new BinaryWriter(fs))
        {
            bw.Write((byte)0x93);
            bw.Write(Encoding.ASCII.GetBytes("NUMPY"));
            bw.Write((byte)1);
            bw.Write((byte)0);

            string dict = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({t}, {j}, 3), }}";
            int preambleLen = 10;
            int padLen = 16 - ((preambleLen + dict.Length + 1) % 16);
            if (padLen == 16) padLen = 0;
            string header = dict + new string(' ', padLen) + "\n";

            byte[] headerBytes = Encoding.ASCII.GetBytes(header);
            bw.Write((ushort)headerBytes.Length);
            bw.Write(headerBytes);

            for (int i = 0; i < data.Count; i++)
                bw.Write(data[i]);
        }
    }

    void WriteDatasetManifest(string characterRoot, string characterName)
    {
        var manifest = new PoseRecord.Data.DatasetManifestData
        {
            updated_at_utc = DateTime.UtcNow.ToString("o"),
            actions = CollectActionEntries(characterRoot, characterName)
        };

        string manifestPath = Path.Combine(characterRoot, "dataset_manifest.json");
        File.WriteAllText(manifestPath, JsonUtility.ToJson(manifest, true), Encoding.UTF8);
    }

    List<PoseRecord.Data.DatasetActionEntry> CollectActionEntries(string characterRoot, string characterName)
    {
        var list = new List<PoseRecord.Data.DatasetActionEntry>();
        if (!Directory.Exists(characterRoot)) return list;

        string sharedCamerasRoot = Path.Combine(characterRoot, "cameras");
        string[] sharedCameraDirs = Directory.Exists(sharedCamerasRoot)
            ? Directory.GetDirectories(sharedCamerasRoot)
            : new string[0];

        foreach (string actionDir in Directory.GetDirectories(characterRoot))
        {
            string actionIdValue = Path.GetFileName(actionDir);
            if (actionIdValue == "cameras") continue;
            if (!Directory.Exists(Path.Combine(actionDir, "frames")) && !Directory.Exists(Path.Combine(actionDir, "kpt2d")))
                continue;

            string actionCamerasRoot = Path.Combine(actionDir, "cameras");
            string[] cameraDirs = Directory.Exists(actionCamerasRoot)
                ? Directory.GetDirectories(actionCamerasRoot)
                : sharedCameraDirs;

            string[] cameraIds = new string[cameraDirs.Length];
            for (int i = 0; i < cameraDirs.Length; i++)
                cameraIds[i] = Path.GetFileName(cameraDirs[i]);

            list.Add(new PoseRecord.Data.DatasetActionEntry
            {
                subject_id = characterName,
                action_id = actionIdValue,
                action_path = $"{characterName}/{actionIdValue}",
                camera_count = cameraIds.Length,
                camera_ids = cameraIds
            });
        }

        return list;
    }

    void FillRecord(PoseRecord.Data.Frame2DRecordData rec, int frameIdx)
    {
        rec.frame_idx = frameIdx;
        rec.time = Time.time - baseTime;
        rec.width = captureWidth;
        rec.height = captureHeight;
        rec.angle_deg = 0f;
        rec.label = GetClipLabel();

        for (int i = 0; i < joints.Length; i++)
        {
            Vector3 vp = cam.WorldToViewportPoint(joints[i].position);
            var jd = rec.joints2d[i];

            jd.conf = (vp.z > 0 && vp.x >= 0 && vp.x <= 1 && vp.y >= 0 && vp.y <= 1) ? 1f : 0f;

            float x = flipX ? 1f - vp.x : vp.x;
            float y = flipYToTopLeft ? 1f - vp.y : vp.y;

            jd.x = x;
            jd.y = y;
        }
    }

    string GetClipLabel()
    {
        if (!targetAnimator) return "NoAnimator";
        var clips = targetAnimator.GetCurrentAnimatorClipInfo(animatorLayer);
        if (clips == null || clips.Length == 0) return "NoClip";
        return clips[0].clip.name;
    }

    void ApplyCameraPresetLikeInspector()
    {
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = Color.black;
        cam.fieldOfView = presetFov;
        cam.nearClipPlane = presetNear;
        cam.farClipPlane = presetFar;
        cam.targetDisplay = Mathf.Max(0, presetTargetDisplay);
        cam.depth = presetDepth;
    }

    static void FlipTextureVertical(Texture2D tex)
    {
        int w = tex.width, h = tex.height;
        Color[] p = tex.GetPixels();
        for (int y = 0; y < h / 2; y++)
        {
            int y2 = h - 1 - y;
            for (int x = 0; x < w; x++)
            {
                int i1 = y * w + x;
                int i2 = y2 * w + x;
                (p[i1], p[i2]) = (p[i2], p[i1]);
            }
        }
        tex.SetPixels(p);
        tex.Apply(false, false);
    }

    void GetAllChildren(Transform t, List<Transform> list)
    {
        foreach (Transform c in t)
        {
            list.Add(c);
            GetAllChildren(c, list);
        }
    }
    // 最小 NPY writer：float32, C-order, shape=(rows, cols)
    static void WriteNpyFloat32_2D(string path, float[] data, int rows, int cols)
    {
        if (data == null) throw new ArgumentNullException(nameof(data));
        if (data.Length != rows * cols) throw new ArgumentException("data length mismatch");

        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");

        using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write))
        using (var bw = new BinaryWriter(fs))
        {
            // magic + version
            bw.Write((byte)0x93);
            bw.Write(Encoding.ASCII.GetBytes("NUMPY"));
            bw.Write((byte)1); // major
            bw.Write((byte)0); // minor

            string header = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}";
            int preamble = 10; // magic(6)+ver(2)+hlen(2)
            int padLen = 16 - ((preamble + header.Length + 1) % 16);
            if (padLen == 16) padLen = 0;
            string fullHeader = header + new string(' ', padLen) + "\n";

            byte[] headerBytes = Encoding.ASCII.GetBytes(fullHeader);
            bw.Write((ushort)headerBytes.Length);
            bw.Write(headerBytes);

            for (int i = 0; i < data.Length; i++)
                bw.Write(data[i]); // little-endian float32
        }
    }
}
