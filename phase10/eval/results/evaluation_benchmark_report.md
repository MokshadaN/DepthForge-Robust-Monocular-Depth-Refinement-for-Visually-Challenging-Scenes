# Step 1: Rigorous Evaluation Report (Raw vs Median-Scale vs Affine-Fitted)

Evaluation comparing Baseline vs Single-Head (Phase 8) vs Multi-Head (Phase 10) across optical edge cases.


### 1. Primary Raw Unaligned Metrics
*Evaluated directly without test-time target fitting or scaling.*

| Domain          | Model                | AbsRel (v)   | RMSE (v)   | SILog (v)  | d1 < 1.25 (^)  |
|-----------------|----------------------|--------------|------------|------------|----------------|
| Overall (All)   | Base MiDaS           | 20.0089       | 0.5526     | 2.0785     | 0.0562         |
| Overall (All)   | Phase 8 SingleHead   | 19.7734       | 0.5434     | 2.0043     | 0.0572         |
| Overall (All)   | Phase 10 MultiHead (Ours) | 18.8425       | 0.5206     | 1.9552     | 0.0581         |
|-----------------|----------------------|--------------|------------|------------|----------------|
| Camo            | Base MiDaS           | 20.4080       | 0.5580     | 2.0776     | 0.0558         |
| Camo            | Phase 8 SingleHead   | 20.1501       | 0.5486     | 2.0084     | 0.0562         |
| Camo            | Phase 10 MultiHead (Ours) | 19.1613       | 0.5248     | 1.9627     | 0.0573         |
|-----------------|----------------------|--------------|------------|------------|----------------|
| Transparent     | Base MiDaS           | 21.3755       | 0.4444     | 2.6830     | 0.0276         |
| Transparent     | Phase 8 SingleHead   | 21.7625       | 0.4414     | 2.4305     | 0.0313         |
| Transparent     | Phase 10 MultiHead (Ours) | 22.3085       | 0.4438     | 2.2198     | 0.0330         |
|-----------------|----------------------|--------------|------------|------------|----------------|
| Textureless     | Base MiDaS           | 3.4770       | 0.4581     | 1.5064     | 0.1012         |
| Textureless     | Phase 8 SingleHead   | 3.4691       | 0.4449     | 1.4260     | 0.1245         |
| Textureless     | Phase 10 MultiHead (Ours) | 3.2624       | 0.4361     | 1.4068     | 0.1173         |
|-----------------|----------------------|--------------|------------|------------|----------------|

### 2. Standard Median Scale-Aligned Metrics
*Normalized using global median scale (standard monocular benchmark protocol).*

| Domain          | Model                | AbsRel (v)   | RMSE (v)   | SILog (v)  | d1 < 1.25 (^)  |
|-----------------|----------------------|--------------|------------|------------|----------------|
| Overall (All)   | Base MiDaS           | 4.8435       | 0.3547     | 1.9714     | 0.1285         |
| Overall (All)   | Phase 8 SingleHead   | 4.4448       | 0.3438     | 1.8891     | 0.1335         |
| Overall (All)   | Phase 10 MultiHead (Ours) | 4.2902       | 0.3389     | 1.8385     | 0.1383         |
|-----------------|----------------------|--------------|------------|------------|----------------|
| Camo            | Base MiDaS           | 4.9154       | 0.3526     | 1.9661     | 0.1297         |
| Camo            | Phase 8 SingleHead   | 4.5092       | 0.3422     | 1.8889     | 0.1346         |
| Camo            | Phase 10 MultiHead (Ours) | 4.3510       | 0.3374     | 1.8425     | 0.1398         |
|-----------------|----------------------|--------------|------------|------------|----------------|
| Transparent     | Base MiDaS           | 1.5442       | 0.2515     | 2.6509     | 0.1156         |
| Transparent     | Phase 8 SingleHead   | 1.4508       | 0.2487     | 2.3678     | 0.1271         |
| Transparent     | Phase 10 MultiHead (Ours) | 1.4252       | 0.2494     | 2.1256     | 0.1242         |
|-----------------|----------------------|--------------|------------|------------|----------------|
| Textureless     | Base MiDaS           | 5.4113       | 0.5344     | 1.4950     | 0.0942         |
| Textureless     | Phase 8 SingleHead   | 4.9926       | 0.4972     | 1.4181     | 0.0966         |
| Textureless     | Phase 10 MultiHead (Ours) | 4.8446       | 0.4872     | 1.3998     | 0.0954         |
|-----------------|----------------------|--------------|------------|------------|----------------|

### 3. Affine-Fitted Least-Squares (Historical Diagnostic)
*Target-fitted affine alignment (s*pred + t). Preserved for audit comparison.*

| Domain          | Model                | AbsRel (v)   | RMSE (v)   | SILog (v)  | d1 < 1.25 (^)  |
|-----------------|----------------------|--------------|------------|------------|----------------|
| Overall (All)   | Base MiDaS           | 2.4125       | 0.1697     | 0.9404     | 0.2244         |
| Overall (All)   | Phase 8 SingleHead   | 2.3733       | 0.1678     | 0.9396     | 0.2290         |
| Overall (All)   | Phase 10 MultiHead (Ours) | 2.3613       | 0.1675     | 0.9383     | 0.2326         |
|-----------------|----------------------|--------------|------------|------------|----------------|
| Camo            | Base MiDaS           | 2.4225       | 0.1687     | 0.9398     | 0.2234         |
| Camo            | Phase 8 SingleHead   | 2.3862       | 0.1667     | 0.9397     | 0.2282         |
| Camo            | Phase 10 MultiHead (Ours) | 2.3686       | 0.1662     | 0.9373     | 0.2322         |
|-----------------|----------------------|--------------|------------|------------|----------------|
| Transparent     | Base MiDaS           | 3.3811       | 0.1917     | 1.1621     | 0.0811         |
| Transparent     | Phase 8 SingleHead   | 3.1950       | 0.1882     | 1.1305     | 0.0819         |
| Transparent     | Phase 10 MultiHead (Ours) | 3.3164       | 0.1922     | 1.1450     | 0.0848         |
|-----------------|----------------------|--------------|------------|------------|----------------|
| Textureless     | Base MiDaS           | 1.0616       | 0.1869     | 0.7423     | 0.4044         |
| Textureless     | Phase 8 SingleHead   | 1.0614       | 0.1875     | 0.7433     | 0.4039         |
| Textureless     | Phase 10 MultiHead (Ours) | 1.1290       | 0.1921     | 0.7688     | 0.3954         |
|-----------------|----------------------|--------------|------------|------------|----------------|