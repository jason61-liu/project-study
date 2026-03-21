package main

import (
	"context"
	"fmt"

	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
	gatewayv1beta1 "sigs.k8s.io/gateway-api/apis/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// GatewayRouteOptions 定义使用 Gateway API 创建路由的选项
type GatewayRouteOptions struct {
	Name             string
	Namespace        string
	GatewayName      string
	GatewayNamespace string
	Host             string
	Path             string
	PathMatchType    *gatewayv1beta1.PathMatchType
	ServiceName      string
	ServicePort      int32
	Headers          map[string]string
}

// CreateGatewayRoute 使用 Gateway API 创建 Higress 路由
func CreateGatewayRoute(k8sClient client.Client, opts GatewayRouteOptions) error {
	// 默认路径匹配类型
	pathMatchType := gatewayv1beta1.PathMatchTypePrefix
	if opts.PathMatchType != nil {
		pathMatchType = *opts.PathMatchType
	}

	// 构建路由规则
	rule := gatewayv1beta1.HTTPRouteRule{
		Matches: []gatewayv1beta1.HTTPRouteMatch{
			{
				Path: &gatewayv1beta1.HTTPPathMatch{
					Type:  &pathMatchType,
					Value: &opts.Path,
				},
			},
		},
		BackendRefs: []gatewayv1beta1.HTTPBackendRef{
			{
				BackendRef: gatewayv1beta1.BackendRef{
					BackendObjectReference: gatewayv1beta1.BackendObjectReference{
						Name: gatewayv1.ObjectName(opts.ServiceName),
						Port: (*gatewayv1.PortNumber)(&opts.ServicePort),
					},
				},
			},
		},
	}

	// 添加头部匹配
	if len(opts.Headers) > 0 {
		headerMatchType := gatewayv1beta1.HeaderMatchTypeExact
		for headerName, headerValue := range opts.Headers {
			rule.Matches[0].Headers = append(rule.Matches[0].Headers, gatewayv1beta1.HTTPHeaderMatch{
				Type:  &headerMatchType,
				Name:  gatewayv1beta1.HTTPHeaderName(headerName),
				Value: headerValue,
			})
		}
	}

	route := &gatewayv1beta1.HTTPRoute{
		ObjectMeta: metav1.ObjectMeta{
			Name:      opts.Name,
			Namespace: opts.Namespace,
		},
		Spec: gatewayv1beta1.HTTPRouteSpec{
			CommonRouteSpec: gatewayv1beta1.CommonRouteSpec{
				ParentRefs: []gatewayv1beta1.ParentReference{
					{
						Name:      gatewayv1.ObjectName(opts.GatewayName),
						Namespace: (*gatewayv1.Namespace)(&opts.GatewayNamespace),
					},
				},
			},
			Hostnames: []gatewayv1beta1.Hostname{gatewayv1beta1.Hostname(opts.Host)},
			Rules:     []gatewayv1beta1.HTTPRouteRule{rule},
		},
	}

	err := k8sClient.Create(context.TODO(), route)
	if err != nil {
		return fmt.Errorf("创建 HTTPRoute 失败: %w", err)
	}

	fmt.Printf("创建的 HTTPRoute: %s/%s\n", route.Namespace, route.Name)
	return nil
}

// CreateGatewayRouteWithMultipleBackends 创建带多个后端的 Gateway 路由（负载均衡）
func CreateGatewayRouteWithMultipleBackends(k8sClient client.Client, opts GatewayRouteOptions, backends []BackendRef) error {
	pathMatchType := gatewayv1beta1.PathMatchTypePrefix
	if opts.PathMatchType != nil {
		pathMatchType = *opts.PathMatchType
	}

	// 构建后端引用列表
	backendRefs := make([]gatewayv1beta1.HTTPBackendRef, len(backends))
	for i, backend := range backends {
		backendRefs[i] = gatewayv1beta1.HTTPBackendRef{
			BackendRef: gatewayv1beta1.BackendRef{
				BackendObjectReference: gatewayv1beta1.BackendObjectReference{
					Name:      gatewayv1.ObjectName(backend.Name),
					Namespace: (*gatewayv1.Namespace)(&backend.Namespace),
					Port:      (*gatewayv1.PortNumber)(&backend.Port),
				},
				Weight: backend.Weight,
			},
		}
	}

	route := &gatewayv1beta1.HTTPRoute{
		ObjectMeta: metav1.ObjectMeta{
			Name:      opts.Name,
			Namespace: opts.Namespace,
		},
		Spec: gatewayv1beta1.HTTPRouteSpec{
			CommonRouteSpec: gatewayv1beta1.CommonRouteSpec{
				ParentRefs: []gatewayv1beta1.ParentReference{
					{
						Name:      gatewayv1.ObjectName(opts.GatewayName),
						Namespace: (*gatewayv1.Namespace)(&opts.GatewayNamespace),
					},
				},
			},
			Hostnames: []gatewayv1beta1.Hostname{gatewayv1beta1.Hostname(opts.Host)},
			Rules: []gatewayv1beta1.HTTPRouteRule{
				{
					Matches: []gatewayv1beta1.HTTPRouteMatch{
						{
							Path: &gatewayv1beta1.HTTPPathMatch{
								Type:  &pathMatchType,
								Value: &opts.Path,
							},
						},
					},
					BackendRefs: backendRefs,
				},
			},
		},
	}

	return k8sClient.Create(context.TODO(), route)
}

// CreateGatewayRouteWithTimeout 创建带超时配置的 Gateway 路由
func CreateGatewayRouteWithTimeout(k8sClient client.Client, opts GatewayRouteOptions, timeout TimeoutConfig) error {
	pathMatchType := gatewayv1beta1.PathMatchTypePrefix
	if opts.PathMatchType != nil {
		pathMatchType = *opts.PathMatchType
	}

	route := &gatewayv1beta1.HTTPRoute{
		ObjectMeta: metav1.ObjectMeta{
			Name:      opts.Name,
			Namespace: opts.Namespace,
		},
		Spec: gatewayv1beta1.HTTPRouteSpec{
			CommonRouteSpec: gatewayv1beta1.CommonRouteSpec{
				ParentRefs: []gatewayv1beta1.ParentReference{
					{
						Name:      gatewayv1.ObjectName(opts.GatewayName),
						Namespace: (*gatewayv1.Namespace)(&opts.GatewayNamespace),
					},
				},
			},
			Hostnames: []gatewayv1beta1.Hostname{gatewayv1beta1.Hostname(opts.Host)},
			Rules: []gatewayv1beta1.HTTPRouteRule{
				{
					Matches: []gatewayv1beta1.HTTPRouteMatch{
						{
							Path: &gatewayv1beta1.HTTPPathMatch{
								Type:  &pathMatchType,
								Value: &opts.Path,
							},
						},
					},
					BackendRefs: []gatewayv1beta1.HTTPBackendRef{
						{
							BackendRef: gatewayv1beta1.BackendRef{
								BackendObjectReference: gatewayv1beta1.BackendObjectReference{
									Name: gatewayv1.ObjectName(opts.ServiceName),
									Port: (*gatewayv1.PortNumber)(&opts.ServicePort),
								},
							},
						},
					},
					Timeouts: &timeout,
				},
			},
		},
	}

	return k8sClient.Create(context.TODO(), route)
}

// BackendRef 后端服务引用
type BackendRef struct {
	Name      string
	Namespace string
	Port      int32
	Weight    *int32
}

// TimeoutConfig 超时配置
type TimeoutConfig = gatewayv1beta1.HTTPRouteTimeouts

// CreateGatewayRouteWithRequestHeaderModifier 创建带请求头修改的 Gateway 路由
func CreateGatewayRouteWithRequestHeaderModifier(k8sClient client.Client, opts GatewayRouteOptions, filters gatewayv1beta1.HTTPRouteFilter) error {
	pathMatchType := gatewayv1beta1.PathMatchTypePrefix
	if opts.PathMatchType != nil {
		pathMatchType = *opts.PathMatchType
	}

	route := &gatewayv1beta1.HTTPRoute{
		ObjectMeta: metav1.ObjectMeta{
			Name:      opts.Name,
			Namespace: opts.Namespace,
		},
		Spec: gatewayv1beta1.HTTPRouteSpec{
			CommonRouteSpec: gatewayv1beta1.CommonRouteSpec{
				ParentRefs: []gatewayv1beta1.ParentReference{
					{
						Name:      gatewayv1.ObjectName(opts.GatewayName),
						Namespace: (*gatewayv1.Namespace)(&opts.GatewayNamespace),
					},
				},
			},
			Hostnames: []gatewayv1beta1.Hostname{gatewayv1beta1.Hostname(opts.Host)},
			Rules: []gatewayv1beta1.HTTPRouteRule{
				{
					Matches: []gatewayv1beta1.HTTPRouteMatch{
						{
							Path: &gatewayv1beta1.HTTPPathMatch{
								Type:  &pathMatchType,
								Value: &opts.Path,
							},
						},
					},
					Filters: []gatewayv1beta1.HTTPRouteFilter{filters},
					BackendRefs: []gatewayv1beta1.HTTPBackendRef{
						{
							BackendRef: gatewayv1beta1.BackendRef{
								BackendObjectReference: gatewayv1beta1.BackendObjectReference{
									Name: gatewayv1.ObjectName(opts.ServiceName),
									Port: (*gatewayv1.PortNumber)(&opts.ServicePort),
								},
							},
						},
					},
				},
			},
		},
	}

	return k8sClient.Create(context.TODO(), route)
}
