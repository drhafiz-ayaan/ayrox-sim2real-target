# Synthetic Data Drone Study

**AyroX Labs: concept study**
Can a drone learn to fly across a campus using only synthetic data?

> Every flight described here happened in simulation. No real flights were used for training or testing.

Video: `VIDEOS/Synthetic_Data_Drone_Study_2026-10-29.mp4` (2:14, 1080p). Not included in this repository: at 123 MB it is over GitHub's 100 MB file limit.

---

## The problem

Training a drone's flight policy runs into two problems:

| Approach | What goes wrong |
|---|---|
| **Train on the real drone** | You get one flight at a time. Every crash costs parts and days. Rare, dangerous situations are the hardest to collect and the most important to learn from. |
| **Train in a clean simulation** | It works in the simulation but fails in the real world. A policy that only saw perfect conditions breaks when the wind, the sensors or the obstacles change. |

## The approach

### 1. Build a digital twin of the real place
**NUST H-12, Islamabad** is modelled at real scale. The terrain, buildings, trees, roads and power lines sit where they really are.

| Size | Buildings | Trees | Also |
|---|---|---|---|
| 3 km × 3 km | 3,070 | 2,888 | Power lines |

A second twin of **UET Taxila** is used only for testing, so the policy is checked on a campus it never trained on.

### 2. Define the task
Fly from a start point to a goal **70–150 m away**, **5 m above the ground**. The drone has to get around buildings, trees and power lines using only its own onboard sensing.

### 3. Generate thousands of varied flights
Each synthetic training flight gets its own:
- obstacle layout (about 8–28 obstacles added per flight)
- wind (about 0.2–1.1 m/s)
- sensor noise (about 4–11 cm)
- drone response

| Synthetic flights | Hours of flight | Obstacle layouts | Real flights |
|---|---|---|---|
| 2,677* | 21.8 | 2,677* | **0** |

\* See [Open questions](#open-questions): elsewhere the video gives the dataset size as 3,000.

### 4. Train the policy and test it on situations it never saw
The trained policy flies itself through obstacle layouts, wind and noise it has never seen. A training run with "corrections" (about 2,400 corrected flights added on top of the varied set) is also tested.

## Results

The video labels these results "Measured, not claimed." Each number is the success rate over 200 test flights on layouts the policy never saw. All training data comes from NUST. The pilot that flew the training flights scores 99–100% on the same tests.

| Training data | NUST, new layouts | UET Taxila, never seen | NUST, harsher than training |
|---|---|---|---|
| 50 clean flights | 58% | 52% | 46% |
| 3,000 clean flights | 72% | 65% | 52% |
| 3,000 varied flights | 89% | 83% | 82% |
| Varied, plus corrections | **92%** | 80% | **88%*** |

\* One frame of the video shows 84% here and a later frame shows 88% (see [Open questions](#open-questions)).

**Main finding: variety matters more than volume.** Going from 50 to 3,000 *clean* flights helps a little. Making the 3,000 flights *varied* helps far more, on every test set, including the campus the policy never saw.

In a side-by-side run with the same layout, wind, start and goal, the policy trained on 50 clean flights crashes. The policy trained on varied synthetic flights reaches the goal.

## Roadmap

| Status | Stage | Description |
|---|---|---|
| ✅ Done | **Simulation study** | Two campus twins and synthetic data, measured on new layouts and on an unseen campus. |
| ⏭ Next | **Hardware tests** | The trained policy on AyroX's hybrid drone, in a controlled space. |
| 🔜 Then | **Campus flights** | Real routes on campus, compared flight for flight with the twin. |

## Open questions

These are inconsistencies in the video that need to be resolved before it is published:

1. **Dataset size:** the stats panel says 2,677 synthetic flights and obstacle layouts. The comparison and results say 3,000 varied flights.
2. **"NUST, harsher than training" score:** "varied, plus corrections" shows 84% in one frame and 88% in another and in the chart.
3. **Corrections at Taxila:** adding corrections lowers the Taxila score (83% → 80%), while the other two test sets improve. Does this reflect corrections collected only at NUST?
4. **Onboard sensing:** the video says "only what the drone itself senses" but doesn't name the sensors.

---

**AyroX**: Simulation-first infrastructure for embodied AI · [ayrox.tech](https://ayrox.tech)
