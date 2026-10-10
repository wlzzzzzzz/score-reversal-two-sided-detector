# Optional upstream code

The main detector and its cached-result replay use the code included in this
repository. The following resources are needed only for additional model or
generation reruns and are excluded by `.gitignore`.

- **SemStamp generation**: obtain [abehou/SemStamp](https://github.com/abehou/SemStamp)
  into `external_code/SemStamp/`. The generator loads `sampling_lsh_utils.py`
  and the upstream sentence embedding dependencies from that checkout. Use
  the matching embedder in `models/stage3_watermarks/SemStamp-c4-sbert/`.
  No redistribution license was present in the inspected upstream tree, so
  its source is not vendored in the public release. The paper's SemStamp
  benchmark texts and detection results are included.
- **AdaDetectGPT**: place its upstream scripts and pretrained witness under
  `external_code/AdaDetectGPT/scripts/`, as required by
  `scripts/run_clean41_strong_baseline_scores.py`. The recorded scores and
  the CPU validation of those scores are included.

Use the source revisions matching the recorded experiment hashes for exact
historical generation; a current upstream checkout alone is not an exact
reproduction of an older modified implementation. The original local SemStamp
snapshot is retained outside the public release.
