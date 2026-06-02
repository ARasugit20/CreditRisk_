# Contributing

## Git commit author (GitHub contributions)

For commits to count on your [GitHub profile](https://github.com/ARasugit20), use an email **verified** on your GitHub account, or your GitHub noreply address:

```text
195702448+ARasugit20@users.noreply.github.com
```

Example (does not change global git config):

```bash
git -c user.email="195702448+ARasugit20@users.noreply.github.com" \
    -c user.name="ARasugit20" \
    commit -m "your message"
```

Add and verify `anola268@asu.edu` at https://github.com/settings/emails if you prefer that address.

## Development

```bash
pip install -r requirements.txt
pytest tests/ -v
ruff check src tests
bash scripts/run_warehouse.sh   # optional: dbt + DuckDB mart
```
