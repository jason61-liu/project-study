package cli

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/shiyiliu/higress-route/internal/validate"
	"github.com/shiyiliu/higress-route/pkg/client"
	"github.com/shiyiliu/higress-route/pkg/route"
)

var (
	createName        string
	createNamespace   string
	createGateway     string
	createGatewayNS   string
	createHost        string
	createPath        string
	createPathMatch   string
	createService     string
	createServicePort int32
	createDryRun      bool
)

// NewCreateCmd creates the create command
func NewCreateCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "create",
		Short: "Create a new HTTPRoute",
		Long: `Create a new HTTPRoute resource for Higress Gateway API.

Example:
  hr create --name my-route --host api.example.com --path /api \
    --service backend --port 8080 --gateway higress-gateway \
    --gateway-namespace higress-system`,
		RunE: runCreate,
	}

	cmd.Flags().StringVarP(&createName, "name", "n", "", "HTTPRoute name (required)")
	cmd.Flags().StringVarP(&createNamespace, "namespace", "N", "default", "Kubernetes namespace")
	cmd.Flags().StringVarP(&createGateway, "gateway", "g", "", "Gateway name (required)")
	cmd.Flags().StringVar(&createGatewayNS, "gateway-namespace", "higress-system", "Gateway namespace")
	cmd.Flags().StringVarP(&createHost, "host", "H", "", "Hostname (required)")
	cmd.Flags().StringVarP(&createPath, "path", "p", "/", "URL path")
	cmd.Flags().StringVar(&createPathMatch, "path-match", "Prefix", "Path match type: Exact, Prefix, or RegularExpression")
	cmd.Flags().StringVarP(&createService, "service", "s", "", "Backend service name (required)")
	cmd.Flags().Int32VarP(&createServicePort, "port", "P", 80, "Backend service port")
	cmd.Flags().BoolVar(&createDryRun, "dry-run", false, "Print the configuration without creating")

	// Mark required flags
	cmd.MarkFlagRequired("name")
	cmd.MarkFlagRequired("gateway")
	cmd.MarkFlagRequired("host")
	cmd.MarkFlagRequired("service")

	return cmd
}

func runCreate(cmd *cobra.Command, args []string) error {
	// Validate input
	if err := validate.ValidateCreateInput(createName, createNamespace, createGateway, createGatewayNS, createHost, createPath, createService, createServicePort); err != nil {
		return fmt.Errorf("validation error: %w", err)
	}

	// Parse path match type
	pathMatchType := route.PathMatchType(createPathMatch)
	if err := validate.ValidatePathMatchType(pathMatchType); err != nil {
		return fmt.Errorf("invalid path match type: %w", err)
	}

	// Build route config
	cfg := route.RouteConfig{
		Name:      createName,
		Namespace: createNamespace,
		Gateway: route.GatewayRef{
			Name:      createGateway,
			Namespace: createGatewayNS,
		},
		Hostname:  createHost,
		Path:     createPath,
		PathMatch: pathMatchType,
		Backend: route.BackendRef{
			Name: createService,
			Port: createServicePort,
		},
	}

	// Dry run - just print the config
	if createDryRun {
		fmt.Println("Dry run - configuration that would be created:")
		fmt.Printf("  Name: %s\n", cfg.Name)
		fmt.Printf("  Namespace: %s\n", cfg.Namespace)
		fmt.Printf("  Gateway: %s/%s\n", cfg.Gateway.Namespace, cfg.Gateway.Name)
		fmt.Printf("  Hostname: %s\n", cfg.Hostname)
		fmt.Printf("  Path: %s (match: %s)\n", cfg.Path, cfg.PathMatch)
		fmt.Printf("  Backend: %s:%d\n", cfg.Backend.Name, cfg.Backend.Port)
		return nil
	}

	// Create Kubernetes client
	k8sClient, err := client.New(client.Options{Kubeconfig: getKubeconfigPath()})
	if err != nil {
		return fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	// Create the HTTPRoute
	if err := route.CreateHTTPRoute(k8sClient, cfg); err != nil {
		return err
	}

	fmt.Println("HTTPRoute created successfully!")
	fmt.Printf("  Name: %s\n", cfg.Name)
	fmt.Printf("  Namespace: %s\n", cfg.Namespace)
	fmt.Printf("  Gateway: %s/%s\n", cfg.Gateway.Namespace, cfg.Gateway.Name)
	fmt.Printf("  Hostname: %s\n", cfg.Hostname)
	fmt.Printf("  Path: %s (match: %s)\n", cfg.Path, cfg.PathMatch)
	fmt.Printf("  Backend: %s:%d\n", cfg.Backend.Name, cfg.Backend.Port)

	return nil
}
