package cli

import (
	"context"
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/spf13/cobra"
	gatewayv1beta1 "sigs.k8s.io/gateway-api/apis/v1beta1"
	"github.com/shiyiliu/higress-route/internal/validate"
	"github.com/shiyiliu/higress-route/pkg/client"
	"github.com/shiyiliu/higress-route/pkg/route"
)

var (
	listNamespace string
	listAllNamespaces bool
)

// NewListCmd creates the list command
func NewListCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List HTTPRoutes",
		Long: `List HTTPRoute resources.

Example:
  hr list --namespace default
  hr list --all-namespaces`,
		RunE: runList,
	}

	cmd.Flags().StringVarP(&listNamespace, "namespace", "N", "default", "Kubernetes namespace")
	cmd.Flags().BoolVarP(&listAllNamespaces, "all-namespaces", "A", false, "List HTTPRoutes in all namespaces")

	return cmd
}

func runList(cmd *cobra.Command, args []string) error {
	// Validate namespace if not using all namespaces
	if !listAllNamespaces {
		if err := validate.ValidateNamespace(listNamespace); err != nil {
			return err
		}
	}

	// Create Kubernetes client
	k8sClient, err := client.New(client.Options{Kubeconfig: getKubeconfigPath()})
	if err != nil {
		return fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	// List HTTPRoutes
	if listAllNamespaces {
		return listAllNamespacesRoutes(k8sClient)
	}
	return listNamespaceRoutes(k8sClient, listNamespace)
}

func listNamespaceRoutes(k8sClient client.Client, namespace string) error {
	routes, err := route.ListHTTPRoutes(k8sClient, namespace)
	if err != nil {
		return err
	}

	if len(routes) == 0 {
		fmt.Printf("No HTTPRoutes found in namespace '%s'\n", namespace)
		return nil
	}

	fmt.Printf("\n=== HTTPRoutes in namespace '%s' ===\n\n", namespace)

	// Use tabwriter for aligned output
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "NAME\tHOSTNAME\tPATH\tBACKEND\tGATEWAY")
	for _, r := range routes {
		backend := getBackendInfo(r)
		gateway := getGatewayInfo(r)
		path := getPathInfo(r)
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n",
			r.Name,
			getHostnameInfo(r),
			path,
			backend,
			gateway,
		)
	}
	w.Flush()

	fmt.Println()
	return nil
}

func listAllNamespacesRoutes(k8sClient client.Client) error {
	// List all routes across all namespaces by listing without namespace filter
	routeList := &gatewayv1beta1.HTTPRouteList{}
	err := k8sClient.List(context.TODO(), routeList)
	if err != nil {
		return fmt.Errorf("failed to list HTTPRoutes: %w", err)
	}

	if len(routeList.Items) == 0 {
		fmt.Println("No HTTPRoutes found")
		return nil
	}

	fmt.Println("\n=== HTTPRoutes across all namespaces ===\n")

	// Use tabwriter for aligned output
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "NAMESPACE\tNAME\tHOSTNAME\tPATH\tBACKEND\tGATEWAY")
	for _, r := range routeList.Items {
		backend := getBackendInfo(r)
		gateway := getGatewayInfo(r)
		path := getPathInfo(r)
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\n",
			r.Namespace,
			r.Name,
			getHostnameInfo(r),
			path,
			backend,
			gateway,
		)
	}
	w.Flush()

	fmt.Println()
	return nil
}

func getHostnameInfo(r gatewayv1beta1.HTTPRoute) string {
	if len(r.Spec.Hostnames) > 0 {
		return string(r.Spec.Hostnames[0])
	}
	return "-"
}

func getPathInfo(r gatewayv1beta1.HTTPRoute) string {
	if len(r.Spec.Rules) > 0 &&
		len(r.Spec.Rules[0].Matches) > 0 &&
		r.Spec.Rules[0].Matches[0].Path != nil {
		return *r.Spec.Rules[0].Matches[0].Path.Value
	}
	return "/"
}

func getBackendInfo(r gatewayv1beta1.HTTPRoute) string {
	if len(r.Spec.Rules) > 0 &&
		len(r.Spec.Rules[0].BackendRefs) > 0 {
		backend := r.Spec.Rules[0].BackendRefs[0]
		name := string(backend.Name)
		if backend.Port != nil {
			return fmt.Sprintf("%s:%d", name, *backend.Port)
		}
		return name
	}
	return "-"
}

func getGatewayInfo(r gatewayv1beta1.HTTPRoute) string {
	if len(r.Spec.ParentRefs) > 0 {
		gateway := r.Spec.ParentRefs[0]
		if gateway.Namespace != nil {
			return fmt.Sprintf("%s/%s", *gateway.Namespace, gateway.Name)
		}
		return string(gateway.Name)
	}
	return "-"
}
