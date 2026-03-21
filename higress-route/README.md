# hr - Higress Gateway API CLI Tool

A command-line tool for managing Higress routes using Kubernetes Gateway API. Higress is a cloud-native API gateway open-sourced by Alibaba, built on Istio and Envoy.

## Features

- **Gateway API Support**: Uses Kubernetes Gateway API (HTTPRoute) for routing
- **Simple CLI**: Create, list, get, and delete HTTPRoutes with ease
- **Path Matching**: Supports Exact, Prefix, and RegularExpression path matching
- **Multiple Backends**: Supports load balancing across multiple backend services
- **Input Validation**: Validates all inputs before creating resources

## Installation

### Build from Source

```bash
cd higress-route
go mod tidy
make build
```

The binary will be created at `bin/hr`.

### Install to GOPATH/bin

```bash
make install
```

### Build for Multiple Platforms

```bash
make build-all
```

This builds binaries for:
- Linux (amd64, arm64)
- macOS (amd64, arm64)
- Windows (amd64)

## Usage

### Create a Route

```bash
# Basic route creation
hr create \
  --name my-route \
  --host api.example.com \
  --path /api \
  --service backend \
  --port 8080 \
  --gateway higress-gateway \
  --gateway-namespace higress-system

# Dry run (preview without creating)
hr create \
  --name my-route \
  --host api.example.com \
  --path /api \
  --service backend \
  --port 8080 \
  --gateway higress-gateway \
  --dry-run

# Custom path match type
hr create \
  --name exact-route \
  --host api.example.com \
  --path /api/v1/users \
  --path-match Exact \
  --service backend \
  --port 8080 \
  --gateway higress-gateway

# Custom namespace
hr create \
  --name my-route \
  --namespace production \
  --host api.example.com \
  --path /api \
  --service backend \
  --port 8080 \
  --gateway higress-gateway
```

### List Routes

```bash
# List routes in default namespace
hr list

# List routes in specific namespace
hr list --namespace production

# List routes in all namespaces
hr list --all-namespaces
```

### Get Route Details

```bash
# Get route details
hr get --name my-route --namespace default

# Get route in default namespace
hr get --name my-route
```

### Delete a Route

```bash
# Delete route
hr delete --name my-route --namespace default

# Delete route in default namespace
hr delete --name my-route
```

### Global Flags

```bash
# Specify custom kubeconfig
hr --kubeconfig /path/to/kubeconfig create --name my-route ...
```

## API Reference

| Command | Flag | Description | Required |
|---------|------|-------------|----------|
| `create` | `--name, -n` | HTTPRoute name | Yes |
| `create` | `--namespace, -N` | Kubernetes namespace | No (default: default) |
| `create` | `--gateway, -g` | Gateway name | Yes |
| `create` | `--gateway-namespace` | Gateway namespace | No (default: higress-system) |
| `create` | `--host, -H` | Hostname | Yes |
| `create` | `--path, -p` | URL path | No (default: /) |
| `create` | `--path-match` | Path match type (Exact, Prefix, RegularExpression) | No (default: Prefix) |
| `create` | `--service, -s` | Backend service name | Yes |
| `create` | `--port, -P` | Backend service port | No (default: 80) |
| `create` | `--dry-run` | Preview without creating | No |
| `delete` | `--name, -n` | HTTPRoute name | Yes |
| `delete` | `--namespace, -N` | Kubernetes namespace | No (default: default) |
| `get` | `--name, -n` | HTTPRoute name | Yes |
| `get` | `--namespace, -N` | Kubernetes namespace | No (default: default) |
| `list` | `--namespace, -N` | Kubernetes namespace | No (default: default) |
| `list` | `--all-namespaces, -A` | List in all namespaces | No |

## Project Structure

```
higress-route/
├── cmd/
│   └── hr/
│       └── main.go              # CLI entry point
├── pkg/
│   ├── client/
│   │   ├── client.go            # Kubernetes client initialization
│   │   └── scheme.go            # Scheme for Gateway API types
│   └── route/
│       ├── route.go             # HTTPRoute operations
│       └── types.go             # Type definitions
├── internal/
│   ├── cli/
│   │   ├── root.go              # Root command
│   │   ├── create.go            # create command
│   │   ├── delete.go            # delete command
│   │   ├── get.go               # get command
│   │   └── list.go              # list command
│   └── validate/
│       └── validate.go          # Input validation
├── go.mod
├── Makefile
└── README.md
```

## Architecture

```
┌─────────────────┐
│     hr CLI      │
│  (cobra +       │
│  controller-    │
│   runtime)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Kubernetes API  │
│   Server        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Higress       │
│  (Gateway       │
│   Controller)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HTTPRoute      │
│   Resource      │
└─────────────────┘
```

## Gateway API vs Ingress API

This tool uses the Gateway API (HTTPRoute) instead of the traditional Ingress API:

| Feature | Gateway API | Ingress API |
|---------|-------------|-------------|
| Specification | Kubernetes standard | Kubernetes standard |
| Resource | HTTPRoute | Ingress |
| Path Matching | Exact, Prefix, RegularExpression | Exact, Prefix, ImplementationSpecific |
| Multiple Backends | Native support | Limited support |
| Header Matching | Rich support | Via annotations |
| Higress Support | Native (v1.2+) | Full support |

## Development

### Run Tests

```bash
make test
```

### Format Code

```bash
make fmt
```

### Lint Code

```bash
make lint
```

### Clean Build Artifacts

```bash
make clean
```

## Example Workflow

```bash
# 1. Create a route
hr create \
  --name api-route \
  --host api.example.com \
  --path /v1 \
  --service api-backend \
  --port 8080 \
  --gateway higress-gateway

# 2. Verify the route was created
hr get --name api-route

# 3. List all routes
hr list

# 4. Delete the route when done
hr delete --name api-route
```

## Requirements

- Go 1.21+
- Access to a Kubernetes cluster with Higress installed
- Valid kubeconfig file

## License

MIT License
