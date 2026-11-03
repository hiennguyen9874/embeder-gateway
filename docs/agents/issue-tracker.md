# Issue tracker: GitHub

Issues and specs live in GitHub Issues. Use the `gh` CLI; it infers the
repository from the Git remote.

## Operations

- Create: `gh issue create --title "..." --body "..."` (use a heredoc for multiline bodies).
- Read: `gh issue view <number> --comments`; also fetch labels when needed.
- List: `gh issue list --state open`, using `--label`, `--state`, and `--json` as needed.
- Comment: `gh issue comment <number> --body "..."`
- Add or remove labels: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`
- Close: `gh issue close <number> --comment "..."`

When a skill says "publish to the issue tracker", create a GitHub issue.
When it says "fetch the relevant ticket", run `gh issue view <number> --comments`.

## Pull requests as a triage surface

**PRs as a request surface: no.** Set to `yes` if external PRs should be
treated as feature requests. When enabled, use the corresponding `gh pr`
commands to read, list, comment on, label, and close PRs. Filter external
authors by `authorAssociation`: `CONTRIBUTOR`, `FIRST_TIME_CONTRIBUTOR`,
or `NONE`. Issues and PRs share a number space; resolve ambiguous numbers
with `gh pr view` and fall back to `gh issue view`.

## Wayfinding operations

A map is an issue labelled `wayfinder:map`, holding Notes,
Decisions-so-far, and Fog. Create child tickets as GitHub sub-issues,
labelled `wayfinder:<type>` (`research`, `prototype`, `grilling`, or
`task`). If sub-issues are unavailable, add children to a task list in
the map and put `Part of #<map>` at the top of each child.

Represent blockers using GitHub's native issue dependencies:
`gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`.
Obtain the blocker's database ID with
`gh api repos/<owner>/<repo>/issues/<n> --jq .id`. If dependencies are
unavailable, use `Blocked by: #<n>, #<n>` in the child body. A ticket is
unblocked when all blockers are closed.

For the frontier, list open children in map order and exclude those with
open blockers or an assignee. Claim the first with
`gh issue edit <n> --add-assignee @me`. Resolve it by commenting the
answer, closing the issue, and appending a gist and link to the map's
Decisions-so-far.
