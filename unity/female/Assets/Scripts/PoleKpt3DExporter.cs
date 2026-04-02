using System;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PoleKpt3DExporter : MonoBehaviour
{
    [Header("Output")]
    [Tooltip("Dataset root folder under project root")]
    public string outRootFolder = "SkiDataset";

    [Tooltip("Character folder name under SkiDataset")]
    public string characterFolderName = "female";

    [Tooltip("Use GameObject name when characterFolderName is empty")]
    public bool autoUseTargetNameAsCharacterFolder = true;

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

    [Header("Pole Keypoints")]
    [Tooltip("Export only pole keypoints (4 points: L-Handle, L-Tip, R-Handle, R-Tip)")]
    public bool autoFindPoleKpts = true;

    [Tooltip("Left pole handle")]
    public Transform poleL_Handle;

    [Tooltip("Left pole tip")]
    public Transform poleL_Tip;

    [Tooltip("Right pole handle")]
    public Transform poleR_Handle;

    [Tooltip("Right pole tip")]
    public Transform poleR_Tip;

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
            Debug.LogError("[PoleKpt3DExporter] targetAnimator is null.");
            yield break;
        }

        // Auto find pole keypoints if needed
        if (autoFindPoleKpts)
            ResolvePoleKpts();

        // Validate pole keypoints
        if (poleL_Handle == null || poleL_Tip == null || poleR_Handle == null || poleR_Tip == null)
        {
            Debug.LogError("[PoleKpt3DExporter] One or more pole keypoints are not assigned. " +
                $"L_Handle={poleL_Handle}, L_Tip={poleL_Tip}, R_Handle={poleR_Handle}, R_Tip={poleR_Tip}");
            yield break;
        }

        var ac = targetAnimator.runtimeAnimatorController;
        if (ac == null)
        {
            Debug.LogError("[PoleKpt3DExporter] RuntimeAnimatorController is null.");
            yield break;
        }

        string datasetRoot = Path.Combine(Application.dataPath, "..", "..", outRootFolder);
        string characterRoot = Path.Combine(datasetRoot, characterFolderName);
        Directory.CreateDirectory(characterRoot);

        var clips = new List<AnimationClip>(ac.animationClips);
        if (clips.Count == 0)
        {
            Debug.LogError("[PoleKpt3DExporter] No clips found in RuntimeAnimatorController.");
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

        Debug.Log("[PoleKpt3DExporter] Export finished.");
    }

    void ResolvePoleKpts()
    {
        // Find pole keypoints by name from the animator's skeleton
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

        if (poleL_Handle == null)
            poleL_Handle = FindByName(targetAnimator.transform, "Pole_L_Handle");
        if (poleL_Tip == null)
            poleL_Tip = FindByName(targetAnimator.transform, "Pole_L_Tip");
        if (poleR_Handle == null)
            poleR_Handle = FindByName(targetAnimator.transform, "Pole_R_Handle");
        if (poleR_Tip == null)
            poleR_Tip = FindByName(targetAnimator.transform, "Pole_R_Tip");
    }

    IEnumerator ExportClipKpt3D(string characterRoot, AnimationClip clip, int clipIndex, int clipCount)
    {
        if (clip == null)
            yield break;

        string safeActionName = MakeSafePathName(clip.name);
        string kpt3dDir = Path.Combine(characterRoot, safeActionName, "kpt3d", "pole");
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

        var joints = new Transform[] { poleL_Handle, poleL_Tip, poleR_Handle, poleR_Tip };
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
            WriteFloatNpy(kpt3dPath, mergedBuffer, sampleFrames.Count, 4, 3);

        if (logClipSummary)
        {
            Debug.Log(
                $"[PoleKpt3DExporter] [{clipIndex + 1}/{clipCount}] action={safeActionName}, " +
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
