import type { AnchorHTMLAttributes, ReactNode } from "react";

import { site } from "@/lib/site";

type Props = Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href"> & {
  children: ReactNode;
};

/**
 * Full-document navigation to Flask `/signin`.
 * Must NOT use React Router — otherwise the SPA would intercept the auth route.
 */
export function DoctorSignInLink({ children, ...rest }: Props) {
  return (
    <a href={site.loginPath} {...rest}>
      {children}
    </a>
  );
}
