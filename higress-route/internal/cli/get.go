package cli

import (
	"context"
	"fmt"
	"time"

	"github.com/spf13/cobra"
	gatewayv1beta1 "sigs.k8s.io/gateway-api/apis/v1beta1"
	"github.com/shiyiliu/higress-route/internal/validate"
	"github.com/shiyiliu/higress-route/pkg/client"
)

var (
	getName      string
	getNamespace string
)

// NewGetCmd creates the get command
func NewGetCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "get",
		Short: "Get details of an HTTPRoute",
		Long: `Get detailed information about an HTTPRoute resource.

Example:
  hr get --name my-route --namespace default`,
		RunE: runGet,
	}

	cmd.Flags().StringVarP(&getName, "name", "n", "", "HTTPRoute name (required)")
	cmd.Flags().StringVarP(&getNamespace, "namespace", "N", "default", "Kubernetes namespace")

	cmd.MarkFlagRequired("name")

	return cmd
}

func runGet(cmd *cobra.Command, args []string) error {
	// Validate input
	if err := validate.ValidateName(getName); err != nil {
		return err
	}

	if err := validate.ValidateNamespace(getNamespace); err != nil {
		return err
	}

	// Create Kubernetes client
	k8sClient, err := client.New(client.Options{Kubeconfig: getKubeconfigPath()})
	if err != nil {
		return fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	// Get the HTTPRoute
	routeKey := client.ObjectKey{Namespace: getNamespace, Name: getName}
	route := &gatewayv1beta1.HTTPRoute{}
	err = k8sClient.Get(context.TODO(), routeKey, route)
	if err != nil {
		return fmt.Errorf("failed to get HTTPRoute: %w", err)
	}

	// Print route details
	printRouteDetails(route)

	return nil
}

func printRouteDetails(r *gatewayv1beta1.HTTPRoute) {
	fmt.Printf("\n=== HTTPRoute: %s/%s ===\n\n", r.Namespace, r.Name)

	// Basic info
	fmt.Println("Basic Info:")
	fmt.Printf("  Namespace:   %s\n", r.Namespace)
	fmt.Printf("  Name:        %s\n", r.Name)
	fmt.Printf("  Created:     %s\n", r.CreationTimestamp.Format(time.RFC3339))
	if r.DeletionTimestamp != nil {
		fmt.Printf("  Deleting:    %s\n", r.DeletionTimestamp.Format(time.RFC3339))
	}

	// Gateway
	fmt.Println("\nGateway:")
	if len(r.Spec.ParentRefs) > 0 {
		for _, p := range r.Spec.ParentRefs {
			if p.Namespace != nil {
				fmt.Printf("  Parent:      %s/%s\n", *p.Namespace, p.Name)
			} else {
				fmt.Printf("  Parent:      %s\n", p.Name)
			}
		}
	} else {
		fmt.Println("  Parent:      (none)")
	}

	// Hostnames
	fmt.Println("\nHostnames:")
	if len(r.Spec.Hostnames) > 0 {
		for _, h := range r.Spec.Hostnames {
			fmt.Printf("  - %s\n", h)
		}
	} else {
		fmt.Println("  (none - matches all)")
	}

	// Rules
	fmt.Println("\nRules:")
	for i, rule := range r.Spec.Rules {
		fmt.Printf("  Rule %d:\n", i+1)

		// Matches
		if len(rule.Matches) > 0 {
			fmt.Println("    Matches:")
			for _, m := range rule.Matches {
				if m.Path != nil {
					matchType := "Prefix"
					if m.Path.Type != nil {
						matchType = string(*m.Path.Type)
					}
					fmt.Printf("      Path: %s (%s)\n", *m.Path.Value, matchType)
				}
				if len(m.Headers) > 0 {
					fmt.Println("      Headers:")
					for _, h := range m.Headers {
						fmt.Printf("        %s: %s\n", h.Name, h.Value)
					}
				}
			}
		}

		// Backends
		if len(rule.BackendRefs) > 0 {
			fmt.Println("    Backends:")
			for _, b := range rule.BackendRefs {
				weight := ""
				if b.Weight != nil {
					weight = fmt.Sprintf(" (weight: %d)", *b.Weight)
				}
				port := ""
				if b.Port != nil {
					port = fmt.Sprintf(":%d", *b.Port)
				}
				fmt.Printf("      - %s%s%s\n", b.Name, port, weight)
			}
		}
	}

	// Labels and Annotations
	fmt.Println("\nMetadata:")
	if len(r.Labels) > 0 {
		fmt.Println("  Labels:")
		for k, v := range r.Labels {
			fmt.Printf("    %s: %s\n", k, v)
		}
	} else {
		fmt.Println("  Labels: (none)")
	}

	if len(r.Annotations) > 0 {
		fmt.Println("  Annotations:")
		for k, v := range r.Annotations {
			fmt.Printf("    %s: %s\n", k, v)
		}
	} else {
		fmt.Println("  Annotations: (none)")
	}

	// Status
	fmt.Println("\nStatus:")
	if r.Status.Parents != nil {
		for _, p := range r.Status.Parents {
			fmt.Printf("  Gateway: %s\n", p.ParentRef.Name)
			if len(p.Conditions) > 0 {
				for _, c := range p.Conditions {
					fmt.Printf("    %s: %s\n", c.Type, c.Reason)
					if c.Message != nil {
						fmt.Printf("      %s\n", *c.Message)
					}
				}
			}
		}
	} else {
		fmt.Println("  No status available")
	}

	fmt.Println()
}
