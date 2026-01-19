# Assets Tools

MCP tool server for 3D asset generation via Meshy API.

## Files

| File | Description |
|------|-------------|
| `meshy.py` | MCP server providing `get_better_assets` tool |
| `meshy_api.py` | Meshy API client library |

## Tools

### meshy.py

- `get_better_assets` - Generate or retrieve 3D assets when existing ones are unsatisfactory

## Configuration

Requires `MESHY_API_KEY` in `utils/_api_keys.py`.

## Environment

Runs in the `agent` conda environment.
