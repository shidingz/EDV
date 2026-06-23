# EDV Algorithm Notes

This document maps the paper-level Execute-Distill-Verify idea into repository
components. The implementation is intentionally small, but the control flow is
the same as a full benchmark system.

## Offline experience construction

```text
Input:
  task q
  heterogeneous executor pool A_exec
  third-party distiller A_distill
  consensus verifier group A_verify = A_exec
  shared memory M_shared
  private memories M_private
  ability matrix B

Execute:
  for each agent a_i in A_exec:
      tau_i = a_i.execute(q)
      B.update(agent=a_i, domain=q.domain, success=tau_i.success)

Distill:
  E_cand = A_distill.compare_and_distill(q, {tau_i})

Verify:
  for each candidate memory e in E_cand:
      votes = {a_j.verify(e, q, {tau_i}) for a_j in A_verify}

      if all votes approve:
          M_shared.add(e)
      else if at least one vote approves:
          for each approving agent a_j:
              M_private[a_j].add(e)
      else:
          discard e
```

## Inference-time usage

```text
Input:
  new task q_test
  executor pool A
  ability matrix B
  shared memory M_shared
  private memories M_private

Select:
  a_star = argmax_a B.score(a, q_test.domain)

Retrieve:
  R = retrieve(M_shared, q_test)
  if R is insufficient:
      R = R union retrieve(M_private[a_star], q_test)

Reason:
  tau = a_star.execute(q_test, memories=R)
  return tau.final_answer
```

## Memory schema

Each memory item stores:

- `title`: short reusable rule name.
- `description`: why the rule matters.
- `content`: concrete action rule.
- `tags`: domain and failure-mode tags used for retrieval.
- `supporting_trajectory_ids`: successful trajectories that support the rule.
- `rejected_trajectory_ids`: failed trajectories that motivate the rule.

## Toy case

The runnable demo uses a translation tool whose schema requires ISO 639-1
language codes. One executor uses full language names and fails. Another checks
the schema and succeeds. EDV distills the contrast into a memory item:

```text
When calling translateWord, pass ISO 639-1 two-letter codes for fromLanguage
and toLanguage.
```

The execution group verifies that this rule is grounded, specific, and useful,
then writes it into the shared memory bank.
