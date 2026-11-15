# glassybarrace

![Night Circuit standings with driver portraits, flags and a live leader card](demo.webp)

```sh
python glassybarrace.py demo.csv race.mov --profiles demo-profiles.csv --title "Night Circuit" --period .8 --fps 24
```

Top 21 by default. Use `--top` to change it and YYYY-MM-DD or YYYY in the first CSV column to count days in first place. Requires matplotlib and FFmpeg.

Optional profiles use `name,portrait,flag,specs`. Image paths are relative to the profiles CSV. Append `#0`, `#1`, etc. to select square portraits or 3:2 flags from a vertical strip.

Fictional drivers. [Flags by Flagpedia](https://flagpedia.net).
