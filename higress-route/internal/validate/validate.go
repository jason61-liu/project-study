package validate

import (
	"fmt"
	"strings"

	"github.com/shiyiliu/higress-route/pkg/route"
)

const (
	// DNS label constraints
	maxNameLength = 253
	minNameLength = 1
)

// ValidateCreateInput validates all input parameters for create command
func ValidateCreateInput(name, namespace, gateway, gatewayNS, host, path, service string, port int32) error {
	if err := ValidateName(name); err != nil {
		return err
	}

	if err := ValidateNamespace(namespace); err != nil {
		return err
	}

	if err := ValidateGatewayName(gateway); err != nil {
		return err
	}

	if err := ValidateNamespace(gatewayNS); err != nil {
		return err
	}

	if err := ValidateHostname(host); err != nil {
		return err
	}

	if err := ValidatePath(path); err != nil {
		return err
	}

	if err := ValidateServiceName(service); err != nil {
		return err
	}

	if err := ValidatePort(port); err != nil {
		return err
	}

	return nil
}

// ValidateName validates a resource name
func ValidateName(name string) error {
	if name == "" {
		return fmt.Errorf("name cannot be empty")
	}

	if len(name) < minNameLength {
		return fmt.Errorf("name must be at least %d character", minNameLength)
	}

	if len(name) > maxNameLength {
		return fmt.Errorf("name must not exceed %d characters", maxNameLength)
	}

	// Check for valid characters (alphanumeric, hyphen, dot)
	// Must start and end with alphanumeric
	if !isValidDNSLabel(name) {
		return fmt.Errorf("name '%s' is not a valid DNS label", name)
	}

	return nil
}

// ValidateNamespace validates a namespace name
func ValidateNamespace(ns string) error {
	if ns == "" {
		return fmt.Errorf("namespace cannot be empty")
	}

	if len(ns) > maxNameLength {
		return fmt.Errorf("namespace must not exceed %d characters", maxNameLength)
	}

	if !isValidDNSLabel(ns) {
		return fmt.Errorf("namespace '%s' is not a valid DNS label", ns)
	}

	return nil
}

// ValidateGatewayName validates a gateway name
func ValidateGatewayName(gateway string) error {
	if gateway == "" {
		return fmt.Errorf("gateway name cannot be empty")
	}

	if !isValidDNSLabel(gateway) {
		return fmt.Errorf("gateway name '%s' is not a valid DNS label", gateway)
	}

	return nil
}

// ValidateHostname validates a hostname
func ValidateHostname(host string) error {
	if host == "" {
		return fmt.Errorf("hostname cannot be empty")
	}

	if len(host) > maxNameLength {
		return fmt.Errorf("hostname must not exceed %d characters", maxNameLength)
	}

	return nil
}

// ValidatePath validates a URL path
func ValidatePath(path string) error {
	if path == "" {
		return fmt.Errorf("path cannot be empty")
	}

	// Path should start with /
	if !strings.HasPrefix(path, "/") {
		return fmt.Errorf("path must start with '/'")
	}

	return nil
}

// ValidateServiceName validates a service name
func ValidateServiceName(service string) error {
	if service == "" {
		return fmt.Errorf("service name cannot be empty")
	}

	if !isValidDNSLabel(service) {
		return fmt.Errorf("service name '%s' is not a valid DNS label", service)
	}

	return nil
}

// ValidatePort validates a port number
func ValidatePort(port int32) error {
	if port < 1 || port > 65535 {
		return fmt.Errorf("port must be between 1 and 65535, got %d", port)
	}
	return nil
}

// ValidatePathMatchType validates the path match type
func ValidatePathMatchType(matchType route.PathMatchType) error {
	switch matchType {
	case route.PathMatchExact, route.PathMatchPrefix, route.PathMatchRegularExpression:
		return nil
	default:
		return fmt.Errorf("invalid path match type: %s (must be Exact, Prefix, or RegularExpression)", matchType)
	}
}

// isValidDNSLabel checks if a string is a valid DNS label
func isValidDNSLabel(s string) bool {
	if s == "" {
		return false
	}

	// Check each character
	for i, r := range s {
		if !isDNSLabelChar(r) {
			return false
		}
		// First and last character cannot be hyphen or dot
		if (i == 0 || i == len(s)-1) && (r == '-' || r == '.') {
			return false
		}
	}

	return true
}

// isDNSLabelChar checks if a rune is valid in a DNS label
func isDNSLabelChar(r rune) bool {
	return (r >= 'a' && r <= 'z') ||
		(r >= 'A' && r <= 'Z') ||
		(r >= '0' && r <= '9') ||
		r == '-' || r == '.'
}
