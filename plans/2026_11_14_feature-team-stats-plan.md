# Team comparison stats (Google-style)

Match overview = Team A | stat | Team B rows, leader bold, split bar per row. Computed client-side from events + player analytics (works local + static).

- [x] Rows: possession, shots, passes, pass accuracy, avg pass length, tackles won, interceptions, ball lost (lower wins), carries, distance, sprints, top speed
- [x] Attribution: tackle/interception -> winning team (1 - loser team_id); shot -> team last in possession
- [x] Player select still shows player cards
- [x] Verified 375 + 1440; Sample republished

## Unresolved
- Shot team inference ok, or tag shots with team in pipeline?
- Add goals/corners later (need detector)?
