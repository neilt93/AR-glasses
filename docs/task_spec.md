# Task Specification: Servo Bracket Assembly

## Task Name
Assemble a servo bracket onto a base plate.

## Overview
The user assembles a small servo/motor bracket onto a base plate using screws
and a screwdriver. This is a constrained, repeatable, visually distinct task
ideal for an MVP assembly-copilot demo.

## Required Objects
| Object         | Count | Notes                        |
|----------------|-------|------------------------------|
| Base plate     | 1     | Flat metal/plastic plate     |
| Servo bracket  | 1     | L-shaped or U-shaped bracket |
| M3 screws      | 2-4   | Small Phillips-head screws   |
| Screwdriver    | 1     | Phillips-head driver         |
| Servo/motor    | 1     | Small hobby servo (optional) |

## Step Sequence

### Step 1 — Place base plate
- **Action**: Place the base plate flat on the work surface.
- **Visual cues**: Base plate visible, flat, centered in frame.
- **Success**: Base plate is stationary and fully visible.

### Step 2 — Pick up bracket
- **Action**: Pick up the servo bracket from the parts area.
- **Visual cues**: Hand reaches toward bracket; bracket leaves surface.
- **Success**: Bracket is in hand or moving toward base plate.

### Step 3 — Align bracket to base plate
- **Action**: Position the bracket so its mounting holes line up with the
  base plate holes.
- **Visual cues**: Bracket overlaps base plate; holes roughly aligned.
- **Success**: Bracket rests on base plate in correct orientation.

### Step 4 — Insert first screw
- **Action**: Place a screw through the first bracket hole into the base plate.
- **Visual cues**: Screw visible near hole; screw head seated in bracket hole.
- **Success**: Screw is inserted (not yet tightened).

### Step 5 — Tighten first screw
- **Action**: Use the screwdriver to tighten the first screw.
- **Visual cues**: Screwdriver engaged with screw; rotational motion.
- **Success**: Screw is flush; bracket does not move when nudged.

### Step 6 — Insert second screw
- **Action**: Place a screw through the second bracket hole.
- **Visual cues**: Second screw visible near second hole.
- **Success**: Screw is inserted into the second hole.

### Step 7 — Tighten second screw
- **Action**: Use the screwdriver to tighten the second screw.
- **Visual cues**: Screwdriver engaged; rotational motion on second screw.
- **Success**: Second screw flush; bracket is firmly attached.

### Step 8 — Verify assembly
- **Action**: Visually and physically check the bracket is secure.
- **Visual cues**: User inspects the assembly; no loose parts.
- **Success**: Bracket is rigid on base plate; all screws tight.

## Common Errors
| Error                       | Detection cue                              |
|-----------------------------|--------------------------------------------|
| Bracket upside down         | Mounting tabs face wrong direction         |
| Wrong screw size            | Screw does not seat flush                  |
| Missed a screw              | Only one screw visible when two expected   |
| Bracket misaligned          | Holes do not overlap with base plate holes |
| Screwdriver slipping        | Repeated failed tightening motions         |

## Environment Constraints (Demo Mode)
- Fixed overhead or chest-mounted camera angle.
- Consistent lighting (desk lamp, no harsh shadows).
- Clean, uncluttered work surface.
- Objects placed in known starting positions.
- One user, two hands visible.

## Success Criteria
- All screws inserted and tightened.
- Bracket firmly attached and correctly oriented.
- Task completed without skipping steps.
