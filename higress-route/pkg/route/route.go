package route

import (
	"context"
	"fmt"

	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
	gatewayv1beta1 "sigs.k8s.io/gateway-api/apis/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// CreateHTTPRoute creates a new HTTPRoute resource
func CreateHTTPRoute(k8sClient client.Client, cfg RouteConfig) error {
	// Convert path match type
	pathMatchType := toGatewayPathMatchType(cfg.PathMatch)

	// Build the backend reference
	backendRef := buildBackendRef(cfg.Backend)

	// Build the route rule
	rule := gatewayv1beta1.HTTPRouteRule{
		Matches: buildMatches(cfg.Path, pathMatchType, cfg.Headers),
		Filters: nil, // Can be extended for filters
		BackendRefs: []gatewayv1beta1.HTTPBackendRef{
			{
				BackendRef: backendRef,
			},
		},
	}

	// Build the HTTPRoute
	route := &gatewayv1beta1.HTTPRoute{
		ObjectMeta: metav1.ObjectMeta{
			Name:        cfg.Name,
			Namespace:   cfg.Namespace,
			Labels:      cfg.Labels,
			Annotations: cfg.Annotations,
		},
		Spec: gatewayv1beta1.HTTPRouteSpec{
			CommonRouteSpec: gatewayv1beta1.CommonRouteSpec{
				ParentRefs: []gatewayv1beta1.ParentReference{
					{
						Name:      gatewayv1.ObjectName(cfg.Gateway.Name),
						Namespace: (*gatewayv1.Namespace)(&cfg.Gateway.Namespace),
					},
				},
			},
			Hostnames: []gatewayv1beta1.Hostname{gatewayv1beta1.Hostname(cfg.Hostname)},
			Rules:     []gatewayv1beta1.HTTPRouteRule{rule},
		},
	}

	// Create the resource
	err := k8sClient.Create(context.TODO(), route)
	if err != nil {
		return fmt.Errorf("failed to create HTTPRoute: %w", err)
	}

	fmt.Printf("HTTPRoute created: %s/%s\n", route.Namespace, route.Name)
	return nil
}

// GetHTTPRoute retrieves an HTTPRoute by name and namespace
func GetHTTPRoute(k8sClient client.Client, namespace, name string) (*gatewayv1beta1.HTTPRoute, error) {
	route := &gatewayv1beta1.HTTPRoute{}
	err := k8sClient.Get(context.TODO(), client.ObjectKey{Namespace: namespace, Name: name}, route)
	if err != nil {
		return nil, fmt.Errorf("failed to get HTTPRoute: %w", err)
	}
	return route, nil
}

// DeleteHTTPRoute deletes an HTTPRoute by name and namespace
func DeleteHTTPRoute(k8sClient client.Client, namespace, name string) error {
	route := &gatewayv1beta1.HTTPRoute{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
	}

	err := k8sClient.Delete(context.TODO(), route)
	if err != nil {
		return fmt.Errorf("failed to delete HTTPRoute: %w", err)
	}

	fmt.Printf("HTTPRoute deleted: %s/%s\n", namespace, name)
	return nil
}

// ListHTTPRoutes lists all HTTPRoutes in a namespace
func ListHTTPRoutes(k8sClient client.Client, namespace string) ([]gatewayv1beta1.HTTPRoute, error) {
	routeList := &gatewayv1beta1.HTTPRouteList{}
	err := k8sClient.List(context.TODO(), routeList, client.InNamespace(namespace))
	if err != nil {
		return nil, fmt.Errorf("failed to list HTTPRoutes: %w", err)
	}
	return routeList.Items, nil
}

// toGatewayPathMatchType converts RouteConfig PathMatchType to Gateway API PathMatchType
func toGatewayPathMatchType(pmt PathMatchType) gatewayv1beta1.PathMatchType {
	switch pmt {
	case PathMatchExact:
		return gatewayv1beta1.PathMatchExact
	case PathMatchRegularExpression:
		return gatewayv1beta1.PathMatchRegularExpression
	default:
		return gatewayv1beta1.PathMatchPrefix
	}
}

// buildBackendRef creates a BackendRef from RouteConfig backend
func buildBackendRef(backend BackendRef) gatewayv1beta1.BackendRef {
	backendRef := gatewayv1beta1.BackendRef{
		BackendObjectReference: gatewayv1beta1.BackendObjectReference{
			Name: gatewayv1.ObjectName(backend.Name),
			Port: (*gatewayv1.PortNumber)(&backend.Port),
		},
	}
	if backend.Weight != nil {
		backendRef.Weight = backend.Weight
	}
	return backendRef
}

// buildMatches creates HTTPRouteMatch slices with path and optional header matching
func buildMatches(path string, pathMatchType gatewayv1beta1.PathMatchType, headers map[string]string) []gatewayv1beta1.HTTPRouteMatch {
	match := gatewayv1beta1.HTTPRouteMatch{
		Path: &gatewayv1beta1.HTTPPathMatch{
			Type:  &pathMatchType,
			Value: &path,
		},
	}

	// Add header matches if provided
	if len(headers) > 0 {
		headerMatchType := gatewayv1beta1.HeaderMatchTypeExact
		for headerName, headerValue := range headers {
			match.Headers = append(match.Headers, gatewayv1beta1.HTTPHeaderMatch{
				Type:  &headerMatchType,
				Name:  gatewayv1beta1.HTTPHeaderName(headerName),
				Value: headerValue,
			})
		}
	}

	return []gatewayv1beta1.HTTPRouteMatch{match}
}
