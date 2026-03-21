package route

// PathMatchType represents the path matching type for HTTPRoute
type PathMatchType string

const (
	// PathMatchExact matches the URL path exactly
	PathMatchExact PathMatchType = "Exact"
	// PathMatchPrefix matches based on a URL path prefix split by /
	PathMatchPrefix PathMatchType = "Prefix"
	// PathMatchRegularExpression matches a URL path against a regular expression
	PathMatchRegularExpression PathMatchType = "RegularExpression"
)

// RouteConfig defines the configuration for creating an HTTPRoute
type RouteConfig struct {
	Name        string
	Namespace   string
	Gateway     GatewayRef
	Hostname    string
	Path        string
	PathMatch   PathMatchType
	Backend     BackendRef
	Headers     map[string]string // Optional: header matching
	Labels      map[string]string
	Annotations map[string]string
}

// GatewayRef represents a reference to a Gateway
type GatewayRef struct {
	Name      string
	Namespace string
}

// BackendRef represents a reference to a backend service
type BackendRef struct {
	Name   string
	Port   int32
	Weight *int32 // Optional: for load balancing
}
