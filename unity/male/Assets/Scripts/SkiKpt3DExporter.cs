using System;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SkiKpt3DExporter : MonoBehaviour
{
    [Header("Output")]
    [Tooltip("Dataset root folder under project root")]
    public string outRootFolder = "SkiDataset";

    [Tooltip("Character folder name under SkiDataset")]
    public string characterFolderName = "male";

    [Tooltip("Overwrite existing 3D kpt npy files")]
    public bool overwriteExistingKpt3d = false;

    [Tooltip("Save per-frame 3D keypoints as frame_XXXXXX.npy")]
    public bool exportPerFrameKpt3d = true;

    [Tooltip("Also save merged clip tensor to kpt3d.npy with shape (T,J,3)")]
    public bool exportMergedKpt3d = false;

    [Tooltip("Per-frame npy filename prefix")]
    private const string perFrameFilePrefix = "frame";

    [Header("Run")]
    public bool autoRunOnPlay = true;
    public float startDelaySec = 0f;

    [Header("Target")]
    public Animator targetAnimator;

    [Header("Ski Keypoints")]
    [Tooltip("Export only ski keypoints (6 points: L-Center, L-Front, L-Back, R-Center, R-Front, R-Back)")]
    public bool autoFindSkiKpts = true;

    [Tooltip("Left ski center")]
    public Transform skiL_Center;

    [Tooltip("Left ski front")]
    public Transform skiL_Front;

    [Tooltip("Left ski back")]
    public Transform skiL_Back;

    [Tooltip("Right ski center")]
    public Transform skiR_Center;

    [Tooltip("Right ski front")]
    public Transform skiR_Front;

    [Tooltip("Right ski back")]
    public Transform skiR_Back;

    [Header("Sampling")]
    [Tooltip("Sample every N clip frames")]
    public int poseEveryNFrames = 1;

    [Tooltip("Append final frame if not aligned with stride")]
    public bool includeLastFrame = true;

    [Tooltip("Log per-clip summary")]
    public bool logClipSummary = true;

    IEnumerator Start()
    {
        if (!autoRunOnPlay)
            yield break;

        if (startDelaySec > 0f)
            yield return new WaitForSeconds(startDelaySec);

        if (targetAnimator == null)
            targetAnimator = GetComponent<Animator>();

        if (targetAnimator == null)
        {
            Debug.LogError("[SkiKpt3DExporter] targetAnimator is null.");
            yield break;
        }

        // Auto find ski keypoints if needed
        if (autoFindSkiKpts)
            ResolveSkiKpts();

        // Validate ski keypoints
        if (skiL_Center == null || skiL_Front == null || skiL_Back == null || 
            skiR_Center == null || skiR_Front == null || skiR_Back == null)
        {
            Debug.LogError("[SkiKpt3DExporter] One or more ski keypoints are not assigned. " +
                $"L_Center={skiL_Center}, L_Front={skiL_Front}, L_Back={skiL_Back}, " +
                $"R_Center={skiR_Center}, R_Front={skiR_Front}, R_Back={skiR_Back}");
            yield break;
        }

        var ac = targetAnimator.runtimeAnimatorController;
        if (ac == null)
        {
            Debug.LogError("[SkiKpt3DExporter] RuntimeAnimatorController is null.");
            yield break;
        }

        string datasetRoot = Path.Combine(Application.dataPath, "..", "..", outRootFolder);
        string characterRoot = Path.Combine(datasetRoot, characterFolderName);
        Directory.CreateDirectory(characterRoot);

        var clips = new List<AnimationClip>(ac.animationClips);
        if (clips.Count == 0)
        {
            Debug.LogError("[SkiKpt3DExporter] No clips found in RuntimeAnimatorController.");
            yield break;
        }

        float oldSpeed = targetAnimator.speed;
        bool oldEnabled = targetAnimator.enabled;

        try
        {
            for (int i = 0; i < clips.Count; i++)
            {
                yield return StartCoroutine(ExportClipKpt3D(characterRoot, clips[i], i, clips.Count));
            }
        }
        finally
        {
            targetAnimator.speed = oldSpeed;
            targetAnimator.enabled = oldEnabled;
        }

        Debug.Log("[SkiKpt3DExporter] Export finished.");
    }

    void ResolveSkiKpts()
    {
        // Find ski keypoints by name from the animator's skeleton
        if (targetAnimator == null)
            return;

        Transform FindByName(Transform root, string name)
        {
            if (root.name == name)
                return root;
            foreach (Transform child in root)
            {
                var result = FindByName(child, name);
                if (result != null)
                    return result;
            }
            return null;
        }

        if (skiL_Center == null)
            skiL_Center = FindByName(targetAnimator.transform, "Ski_L_center");
        if (skiL_Front == null)
            skiL_Front = FindByName(targetAnimator.transform, "Ski_L_front");
        if (skiL_Back == null)
            skiL_Back = FindByName(targetAnimator.transform, "Ski_L_back");
        if (skiR_Center == null)
            skiR_Center = FindByName(targetAnimator.transform, "Ski_R_center");
        if (skiR_Front == null)
            skiR_Front = FindByName(targetAnimator.transform, "Ski_R_front");
        if (skiR_Back == null)
            skiR_Back = FindByName(targetAnimator.transform, "Ski_R_back");
    }

    IEnumerator ExportClipKpt3D(string characterRoot, AnimationClip clip, int clipIndex, int clipCount)
    {
        if (clip == null)
            yield break;

        string safeActionName = MakeSafePathName(clip.name);
        string kpt3dDir = Path.Combine(characterRoot, safeActionName, "kpt3d", "ski");
        Directory.CreateDirectory(kpt3dDir);
        string kpt3dPath = Path.Combine(kpt3dDir, "kpt3d.npy");

        int frameCount = Mathf.Max(1, Mathf.RoundToInt(clip.length * clip.frameRate));
        int stride = Mathf.Max(1, poseEveryNFrames);

        var sampleFrames = new List<int>(frameCount / stride + 2);
        for (int f = 0; f < frameCount; f += stride)
            sampleFrames.Add(f);

        int lastFrame = Mathf.Max(0, frameCount - 1);
        if (includeLastFrame && (sampleFrames.Count == 0 || sampleFrames[sampleFrames.Count - 1] != lastFrame))
            sampleFrames.Add(lastFrame);

        var joints = new Transform[] { skiL_Center, skiL_Front, skiL_Back, skiR_Center, skiR_Front, skiR_Back };
        var mergedBuffer = exportMergedKpt3d ? new List<float>(sampleFrames.Count * joints.Length * 3) : null;
        int savedPerFrameCount = 0;

        for (int s = 0; s < sampleFrames.Count; s++)
        {
            int localFrame = sampleFrames[s];
            float denom = Mathf.Max(1f, frameCount - 1f);
            float t01 = Mathf.Clamp01(localFrame / denom);
            float tSec = t01 * clip.length;

            targetAnimator.enabled = false;
            clip.SampleAnimation(targetAnimator.gameObject, tSec);

            var frameBuffer = new List<float>(joints.Length * 3);
            for (int i = 0; i < joints.Length; i++)
            {
                var t = joints[i];
                Vector3 p = t != null ? t.position : Vector3.zero;
                frameBuffer.Add(p.x);
                frameBuffer.Add(p.y);
                frameBuffer.Add(p.z);
            }

            if (exportPerFrameKpt3d)
            {
                string framePath = Path.Combine(kpt3dDir, $"{perFrameFilePrefix}_{s:D6}.npy");
                if (!File.Exists(framePath) || overwriteExistingKpt3d)
                {
                    WriteFloatNpy(framePath, frameBuffer, joints.Length, 3);
                    savedPerFrameCount++;
                }
            }

            if (mergedBuffer != null)
                mergedBuffer.AddRange(frameBuffer);

            // Avoid freezing editor on very long clips.
            if ((s & 63) == 0)
                yield return null;
        }

        if (mergedBuffer != null && (!File.Exists(kpt3dPath) || overwriteExistingKpt3d))
            WriteFloatNpy(kpt3dPath, mergedBuffer, sampleFrames.Count, 6, 3);

        if (logClipSummary)
        {
            Debug.Log(
                $"[SkiKpt3DExporter] [{clipIndex + 1}/{clipCount}] action={safeActionName}, " +
                $"clipLen={clip.length:F3}s, clipFps={clip.frameRate:F2}, clipFrames={frameCount}, " +
                $"sampledFrames={sampleFrames.Count}, savedPerFrame={savedPerFrameCount}, " +
                $"merged={(exportMergedKpt3d ? "yes" : "no")}");
        }
    }

    void WriteFloatNpy(string path, List<float> data, int d0, int d1, int d2)
    {
        using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write))
        using (var bw = new BinaryWriter(fs))
        {
            bw.Write((byte)0x93);
            bw.Write(Encoding.ASCII.GetBytes("NUMPY"));
            bw.Write((byte)1);
            bw.Write((byte)0);

            string dict = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({d0}, {d1}, {d2}), }}";
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

    void WriteFloatNpy(string path, List<float> data, int d0, int d1)
    {
        using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write))
        using (var bw = new BinaryWriter(fs))
        {
            bw.Write((byte)0x93);
            bw.Write(Encoding.ASCII.GetBytes("NUMPY"));
            bw.Write((byte)1);
            bw.Write((byte)0);

            string dict = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({d0}, {d1}), }}";
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

    string MakeSafePathName(string name)
    {
        string safe = string.IsNullOrWhiteSpace(name) ? "UnknownAction" : name;
        char[] invalid = Path.GetInvalidFileNameChars();
        for (int i = 0; i < invalid.Length; i++)
            safe = safe.Replace(invalid[i], '_');
        return safe;
    }
}
